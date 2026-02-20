import json
import requests
import time
import random
import sys

# Configuration
API_URL = "http://localhost:8000/v1/predict"
SOURCE_FILE = "../data/stream_source.json"

def simulate_traffic():
    print(f"Loading live transaction stream from {SOURCE_FILE}...")
    try:
        with open(SOURCE_FILE, 'r') as f:
            transactions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {SOURCE_FILE} not found. Run 'src/split_and_upload.py' first.")
        sys.exit(1)
        
    print(f"Loaded {len(transactions)} transactions to replay.")
    print(f"Targeting API: {API_URL}")
    print("Starting simulation (Press Ctrl+C to stop)...\n")
    
    success_count = 0
    error_count = 0
    
    try:
        for i, txn in enumerate(transactions):
            # Prepare payload (API expects specific fields)
            payload = {
                "transaction_id": txn.get('transaction_id'),
                "cashier_id": txn.get('cashier_id'),
                "item_count": txn.get('item_count'),
                "total_amount": txn.get('total_amount'),
                "void_count": txn.get('void_count'),
                "no_sale_count": txn.get('no_sale_count'),
                "weight_variance": txn.get('weight_variance'),
                # For simulation, we can pass 0 or pre-calculated rolling values if available
                # In real life, the Feature Store takes care of this.
                "rolling_void_count": 0, 
                "rolling_no_sale_rate": 0
            }
            
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[{i+1}/{len(transactions)}] {txn['transaction_id']} -> Score: {result['risk_score']} ({result['risk_level']}) | Latency: {result['latency_ms']}ms")
                    success_count += 1
                else:
                    print(f"[{i+1}/{len(transactions)}] {txn['transaction_id']} -> Error {response.status_code}: {response.text}")
                    error_count += 1
                    
            except requests.exceptions.ConnectionError:
                print(f"[{i+1}/{len(transactions)}] ❌ API Connection Failed. Is the server running?")
                error_count += 1
                time.sleep(1) # Wait a bit before retrying
                continue

            # Simulate variable traffic interval (50ms to 500ms)
            sleep_ms = random.randint(50, 500)
            time.sleep(sleep_ms / 1000.0)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        
    print("\n--- Simulation Summary ---")
    print(f"Total Requests: {success_count + error_count}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    simulate_traffic()
