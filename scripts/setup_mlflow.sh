#!/bin/bash
# setup_mlflow.sh
# Script to setup MLflow Tracking Server on EC2
# Usage: ./setup_mlflow.sh <S3_BUCKET_URI>

S3_BUCKET=$1

if [ -z "$S3_BUCKET" ]; then
    echo "Usage: ./setup_mlflow.sh <S3_BUCKET_URI>"
    exit 1
fi

echo "Setting up MLflow Server..."

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install MLflow and AWS CLI dependencies
pip install mlflow boto3

# Create local storage for SQLite backend
mkdir -p mlruns

# Create systemd service for MLflow
echo "Creating systemd service..."
sudo bash -c 'cat > /etc/systemd/system/mlflow.service <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/venv/bin/mlflow server \
    --backend-store-uri sqlite:///home/ubuntu/mlruns/mlflow.db \
    --default-artifact-root '$S3_BUCKET' \
    --host 0.0.0.0 \
    --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF'

# Reload systemd and start MLflow
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow

echo "MLflow server started as a service."
echo "Status:"
sudo systemctl status mlflow --no-pager
