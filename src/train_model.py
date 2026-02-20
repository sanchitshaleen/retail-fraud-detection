import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import boto3
import io

# Configuration
BUCKET_NAME = "retail-fraud-artifacts-1771605646"
S3_FEATURES_KEY = "features/train_features.csv"

# --- 1. Load Processed Data (From S3) ---
print(f"Loading processed features from s3://{BUCKET_NAME}/{S3_FEATURES_KEY}...")
s3 = boto3.client('s3')

try:
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_FEATURES_KEY)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
except Exception as e:
    print(f"Error loading from S3: {e}")
    exit(1)

# Select features for modeling
features = [
    'item_count', 'total_amount', 'void_count', 'no_sale_count', 'weight_variance',
    'rolling_void_count', 'rolling_no_sale_rate', 'void_item_ratio', 'weight_variance_zscore'
]
target = 'is_fraud'

print(f"Features: {features}")

# --- 2. Dual-Model Training ---

X = df[features]
y = df[target]

# Split for Supervised Learning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Model 1: XGBoost (Supervised)
print("\nTraining XGBoost...")
# Calculate scale_pos_weight
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {ratio:.2f}")

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='aucpr' # Optimize for Precision-Recall Area
)
xgb_model.fit(X_train, y_train)

# Predictions
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- XGBoost Evaluation ---")
print(classification_report(y_test, (y_prob_xgb > 0.5).astype(int)))
ap_score = average_precision_score(y_test, y_prob_xgb)
print(f"Average Precision Score: {ap_score:.4f}")

# Model 2: Isolation Forest (Unsupervised)
print("\nTraining Isolation Forest...")
# Train on the entire *training* set (X_train) to learn "normal"
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.015, # We know roughly 1.5% are peculiar
    random_state=42,
    n_jobs=-1
)
iso_model.fit(X_train)

# Predict on test
iso_scores_raw = iso_model.decision_function(X_test) 

# Normalize Isolation Forest scores to 0-1 range for risk calculation
# Invert so high = anomalous
iso_risk_score = 1 - ((iso_scores_raw - iso_scores_raw.min()) / (iso_scores_raw.max() - iso_scores_raw.min()))

# --- 3. Hybrid Ensembling Logic ---

def calculate_integrity_score(xgb_prob, iso_risk):
    """
    Combines XGBoost probability (Known Fraud) and Isolation Forest Risk (Unknown Anomaly).
    Weighted Average: 70% XGB, 30% IsoForest.
     Integrity Score = 100 * (1 - Total_Risk)
    """
    w_xgb = 0.7
    w_iso = 0.3
    
    total_risk = (w_xgb * xgb_prob) + (w_iso * iso_risk)
    total_risk = np.clip(total_risk, 0, 1)
    
    integrity_score = 100 * (1 - total_risk)
    return integrity_score

print("\nCalculating Integrity Scores...")
integrity_scores = calculate_integrity_score(y_prob_xgb, iso_risk_score)

# Add to test dataframe for inspection
test_df = X_test.copy()
test_df['is_fraud_true'] = y_test
test_df['xgb_prob'] = y_prob_xgb
test_df['iso_scores_raw'] = iso_scores_raw
test_df['iso_risk'] = iso_risk_score
test_df['integrity_score'] = integrity_scores

# --- 4. Inspect Results ---
print("\n--- Integrity Score Stats ---")
print(test_df.groupby('is_fraud_true')['integrity_score'].describe())

# Check High Risk cases
print("\n--- Low Integrity Examples (Score < 50) ---")
low_integrity = test_df[test_df['integrity_score'] < 50].head(5)
print(low_integrity[['xgb_prob', 'iso_risk', 'integrity_score', 'is_fraud_true']])

# Check "Anomalous Outliers"
print("\n--- Anomalous Outliers Detection Check (is_fraud=0 but high iso_risk) ---")
anomalies = test_df[(test_df['is_fraud_true'] == 0) & (test_df['iso_risk'] > 0.8)]
if not anomalies.empty:
    print(f"Found {len(anomalies)} non-fraud anomalies detected by Isolation Forest.")
    print(anomalies[['total_amount', 'item_count', 'xgb_prob', 'iso_risk', 'integrity_score']].head())
else:
    print("No high-confidence non-fraud anomalies found in test set.")


import mlflow
import mlflow.sklearn
import mlflow.xgboost

# --- 5. MLflow Logging & Registration ---
mlflow.set_tracking_uri("http://localhost:5000") # Default, override with env var in prod
mlflow.set_experiment("retail-fraud-experiment")

with mlflow.start_run():
    # Log Params
    mlflow.log_params({
        "xgb_estimators": 100,
        "xgb_max_depth": 6,
        "xgb_learning_rate": 0.1,
        "iso_contamination": 0.015,
        "class_imbalance_ratio": ratio
    })

    # Log Metrics
    mlflow.log_metrics({
        "average_precision_score": ap_score,
        "mean_integrity_score_fraud": test_df[test_df['is_fraud_true']==1]['integrity_score'].mean(),
        "mean_integrity_score_legit": test_df[test_df['is_fraud_true']==0]['integrity_score'].mean()
    })

    # Log Artifacts (Plots)
    mlflow.log_artifact('../plots/integrity_score_dist.png')
    mlflow.log_artifact('../plots/xgb_pr_curve.png')

    # Log Models (and Register)
    print("\nLogging models to MLflow...")
    mlflow.xgboost.log_model(
        xgb_model, 
        "model_xgb", 
        registered_model_name="retail-fraud-xgb"
    )
    
    # Log Isolation Forest
    mlflow.sklearn.log_model(
        iso_model, 
        "model_iso",
        registered_model_name="retail-fraud-iso"
    )
    
    # Save locally for local testing/dev
    print("Saving models locally to ../app/...")
    joblib.dump(xgb_model, '../app/model_xgb.joblib')
    joblib.dump(iso_model, '../app/model_iso.joblib')

print("Training Run Complete. Models logged to MLflow.")
