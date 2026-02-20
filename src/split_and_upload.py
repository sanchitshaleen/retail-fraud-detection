import pandas as pd
import boto3
import json
import os
import sys

# Configuration
BUCKET_NAME = "retail-fraud-artifacts-1771605646"
RAW_DATA_PATH = "../data/retail_pos_data.csv"
S3_TRAIN_KEY = "raw/train_data.csv"
STREAM_SOURCE_PATH = "../data/stream_source.json"

def split_and_upload():
    print(f"Loading data from {RAW_DATA_PATH}...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found. Run generate_data.py first.")
        sys.exit(1)

    df = pd.read_csv(RAW_DATA_PATH)
    
    # Sort by timestamp to simulate realistic time split
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
    # Split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    stream_df = df.iloc[split_idx:]
    
    print(f"Total Rows: {len(df)}")
    print(f"Historical (Training) Rows: {len(train_df)}")
    print(f"Live (Streaming) Rows: {len(stream_df)}")
    
    # 1. Upload Historical to S3
    print(f"\nUploading Historical Data to s3://{BUCKET_NAME}/{S3_TRAIN_KEY}...")
    
    # Save to temp csv
    temp_csv = "temp_train.csv"
    train_df.to_csv(temp_csv, index=False)
    
    s3 = boto3.client('s3')
    try:
        s3.upload_file(temp_csv, BUCKET_NAME, S3_TRAIN_KEY)
        print("✅ Upload Successful.")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        print("Ensure you have AWS credentials configured and the bucket exists.")
        # Fallback for demo if no AWS creds: save locally to simulate S3
        # train_df.to_csv(f"../data/s3_simulated_train.csv", index=False)
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

    # 2. Save Live Data for Simulation
    print(f"\nSaving Live Data to {STREAM_SOURCE_PATH}...")
    # Convert datetime to string for JSON serialization
    if 'timestamp' in stream_df.columns:
        stream_df['timestamp'] = stream_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
    records = stream_df.to_dict(orient='records')
    with open(STREAM_SOURCE_PATH, 'w') as f:
        json.dump(records, f, indent=2)
    
    print("✅ Live Data Saved.")
    print("\nData Architecture Simulation Prep Complete.")

if __name__ == "__main__":
    split_and_upload()
