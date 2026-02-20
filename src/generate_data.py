import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_retail_data(num_rows=50000):
    np.random.seed(42)
    random.seed(42)

    # --- 1. Base Data Generation (Legitimate - 98%) ---
    n_legit = int(num_rows * 0.98)
    
    # Cashier IDs
    cashier_ids = [f'C{str(i).zfill(3)}' for i in range(1, 51)]
    
    # Timestamps (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(seconds=random.randint(0, 30*24*60*60)) for _ in range(num_rows)]
    timestamps.sort()
    
    data = {
        'transaction_id': [f'T{str(i).zfill(6)}' for i in range(1, num_rows + 1)],
        'cashier_id': np.random.choice(cashier_ids, num_rows),
        'timestamp': timestamps,
        'item_count': np.random.randint(1, 51, num_rows),
        'total_amount': np.random.uniform(5.0, 500.0, num_rows).round(2),
        'void_count': np.random.randint(0, 3, num_rows), # Mostly 0-2
        'no_sale_count': np.random.choice([0, 1], num_rows, p=[0.99, 0.01]), # Mostly 0
        'weight_variance': np.random.normal(0, 5, num_rows).round(2), # Gaussian noise around 0
        'is_fraud': np.zeros(num_rows, dtype=int)
    }
    
    df = pd.DataFrame(data)
    
    # --- 2. Inject Fraud Profiles ---
    
    # A. Sweethearting (1% of data -> ~500 rows)
    # High weight_variance (> 500g), Low total_amount
    n_sweetheart = int(num_rows * 0.01)
    sweetheart_indices = np.random.choice(df.index, n_sweetheart, replace=False)
    
    df.loc[sweetheart_indices, 'weight_variance'] = np.random.uniform(500, 1500, n_sweetheart).round(2)
    df.loc[sweetheart_indices, 'total_amount'] = np.random.uniform(1.0, 20.0, n_sweetheart).round(2) # Low amount
    df.loc[sweetheart_indices, 'is_fraud'] = 1
    
    # B. Cash Skimming (0.5% of data -> ~250 rows)
    # High no_sale_count, High void_count relative to item_count
    # We'll pick from non-sweetheart indices to avoid overlapping fraud types on same rows for clarity, 
    # though overlap is possible in real life.
    remaining_indices = df.index.difference(sweetheart_indices)
    n_skimming = int(num_rows * 0.005)
    skimming_indices = np.random.choice(remaining_indices, n_skimming, replace=False)
    
    df.loc[skimming_indices, 'no_sale_count'] = 1
    # Make void_count high, close to item_count
    df.loc[skimming_indices, 'item_count'] = np.random.randint(5, 20, n_skimming)
    df.loc[skimming_indices, 'void_count'] = df.loc[skimming_indices, 'item_count'] - np.random.randint(1, 3, n_skimming) 
    df.loc[skimming_indices, 'is_fraud'] = 1

    # --- 3. Inject Anomalous Outliers (0.5% of data -> ~250 rows) ---
    # Unlabeled (is_fraud = 0), but extreme values.
    remaining_indices = df.index.difference(sweetheart_indices).difference(skimming_indices)
    n_outliers = int(num_rows * 0.005)
    outlier_indices = np.random.choice(remaining_indices, n_outliers, replace=False)
    
    # Mix of extreme item counts or extreme amounts
    # Split outliers into two groups
    n_extreme_items = n_outliers // 2
    n_extreme_amount = n_outliers - n_extreme_items
    
    idx_items = outlier_indices[:n_extreme_items]
    idx_amount = outlier_indices[n_extreme_items:]
    
    df.loc[idx_items, 'item_count'] = np.random.randint(300, 1000, n_extreme_items)
    df.loc[idx_items, 'total_amount'] = np.random.uniform(5000, 20000, n_extreme_items).round(2)
    
    df.loc[idx_amount, 'total_amount'] = np.random.uniform(10000, 50000, n_extreme_amount).round(2)
    
    # Ensure these are NOT labeled as fraud for the supervised model
    df.loc[outlier_indices, 'is_fraud'] = 0 
    
    return df

if __name__ == "__main__":
    print("Generating synthetic retail data...")
    df = generate_retail_data(50000)
    
    print(f"Data generated: {df.shape}")
    
    # Save to CSV
    filename = "../data/retail_pos_data.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")
    
    # --- Validation Summary ---
    print("\n--- Value Counts for 'is_fraud' ---")
    print(df['is_fraud'].value_counts(normalize=True))
    print(df['is_fraud'].value_counts())
    
    print("\n--- Mean Weight Variance by Group ---")
    # Group by fraud status for simple check
    print(df.groupby('is_fraud')['weight_variance'].mean())
    
    print("\n--- Outlier Check (Max Values) ---")
    print(df[['item_count', 'total_amount', 'weight_variance']].max())
