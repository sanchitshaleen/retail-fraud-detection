import pandas as pd
import numpy as np
import joblib
import boto3
import io
import sys

# Configuration
BUCKET_NAME = "retail-fraud-artifacts-1771605646"
S3_RAW_KEY = "raw/train_data.csv"
S3_FEATURES_KEY = "features/train_features.csv"
S3_STATS_KEY = "artifacts/cashier_stats.joblib"

def feature_engineering():
    print(f"Loading raw data from s3://{BUCKET_NAME}/{S3_RAW_KEY}...")
    s3 = boto3.client('s3')
    
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=S3_RAW_KEY)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        print(f"Error loading from S3: {e}")
        print("Ensure 'src/split_and_upload.py' has been run and AWS creds are set.")
        sys.exit(1)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("Sorting data...")
    # Sort by cashier and time for rolling calculations
    df = df.sort_values(by=['cashier_id', 'timestamp'])

    # --- Feature Engineering ---
    print("Calculating rolling features...")
    # A. Rolling Windows (60-minute)
    df_indexed = df.set_index('timestamp')
    grouped = df_indexed.groupby('cashier_id')

    # Rolling counts/sums
    df['rolling_void_count'] = grouped['void_count'].rolling('1h').sum().values
    df['rolling_no_sale_count'] = grouped['no_sale_count'].rolling('1h').sum().values
    df['rolling_txn_count'] = grouped['transaction_id'].rolling('1h').count().values

    # Avoid division by zero for rates
    df['rolling_no_sale_rate'] = df['rolling_no_sale_count'] / df['rolling_txn_count']

    # B. Ratios
    print("Calculating ratios...")
    df['void_item_ratio'] = df['void_count'] / df['item_count']

    # C. Statistical Deviations (Z-score of weight_variance)
    print("Calculating statistical deviations...")
    # Calculate historical mean/std per cashier
    # We save this artifact because we'll need it for Inference (Phase 3) to score new transactions against history
    cashier_stats = df.groupby('cashier_id')['weight_variance'].agg(['mean', 'std']).reset_index()
    cashier_stats.columns = ['cashier_id', 'cashier_mean_weight_var', 'cashier_std_weight_var']
    
    # Save cashier stats for production inference
    # Select final features + target + metadata
    final_cols = [
        'transaction_id', 'cashier_id', 'timestamp', 'is_fraud',
        'item_count', 'total_amount', 'void_count', 'no_sale_count', 'weight_variance',
        'rolling_void_count', 'rolling_no_sale_count', 'rolling_no_sale_rate', 'void_item_ratio', 'weight_variance_zscore'
    ]
    
    # Ensure all columns exist before selecting
    for col in final_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} missing from dataframe")
            # Create if missing (e.g. rolling_no_sale_rate)
            df[col] = 0

    df_final = df[final_cols]
    
    # Save Features to S3
    print(f"Saving features to s3://{BUCKET_NAME}/{S3_FEATURES_KEY}...")
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=S3_FEATURES_KEY, Body=csv_buffer.getvalue())
    print("Features saved to S3.")
    
    # Save Cashier Stats (Artifact)
    # We also save this locally for 'app/main.py' to use
    print(f"Saving stats artifact to ../app/cashier_stats.joblib and S3...")
    joblib.dump(cashier_stats, '../app/cashier_stats.joblib')
    
    # Upload stats to S3
    with open('../app/cashier_stats.joblib', 'rb') as f:
        s3.put_object(Bucket=BUCKET_NAME, Key=S3_STATS_KEY, Body=f)

    print("Feature Engineering Complete.")

if __name__ == "__main__":
    feature_engineering()
