#!/bin/bash
# scripts/setup_aws_infra.sh
# Automates the creation of core AWS resources for Retail Fraud Detection system.
# USAGE: ./scripts/setup_aws_infra.sh <AWS_REGION> [UNIQUE_SUFFIX]

REGION=${1:-us-east-1}
SUFFIX=${2:-$(date +%s)} # Default to timestamp if not provided

# Resource Names
BUCKET_NAME="retail-fraud-artifacts-${SUFFIX}"
REPO_NAME="retail-fraud-detection"
CLUSTER_NAME="retail-fraud-cluster"
SG_NAME="retail-fraud-sg"
EXECUTION_ROLE_NAME="ecsTaskExecutionRole"

echo "=============================================="
echo "Starting AWS Infrastructure Setup in $REGION"
echo "=============================================="

# 1. Create S3 Bucket (MLflow Artifacts)
echo "[1/5] Creating S3 Bucket: $BUCKET_NAME..."
if [ "$REGION" == "us-east-1" ]; then
  aws s3api create-bucket --bucket $BUCKET_NAME --region $REGION
else
  aws s3api create-bucket --bucket $BUCKET_NAME --region $REGION --create-bucket-configuration LocationConstraint=$REGION
fi
echo "✅ S3 Bucket created: s3://$BUCKET_NAME"

# 2. Create ECR Repository
echo "[2/5] Creating ECR Repository: $REPO_NAME..."
aws ecr create-repository --repository-name $REPO_NAME --region $REGION || echo "Repo likely exists, skipping..."
REPO_URI=$(aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION --query 'repositories[0].repositoryUri' --output text)
echo "✅ ECR Repo created: $REPO_URI"

# 3. Create Lambda Execution Role
ROLE_NAME="RetailFraudLambdaRole"
echo "[3/3] Checking IAM Role for Lambda: $ROLE_NAME..."

aws iam get-role --role-name $ROLE_NAME >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ IAM Role '$ROLE_NAME' exists."
    ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
else
    echo "⚠️  IAM Role '$ROLE_NAME' not found. Creating..."
    # Create trust policy for Lambda
    cat > lambda-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
    ROLE_ARN=$(aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://lambda-trust-policy.json --query 'Role.Arn' --output text)
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    rm lambda-trust-policy.json
    echo "✅ IAM Role '$ROLE_NAME' created."
    # Wait for IAM role to propagate
    echo "Waiting 10 seconds for IAM role to propagate..."
    sleep 10
fi

echo "=============================================="
echo "SETUP COMPLETE (Lambda Strategy)!"
echo "=============================================="
echo "Please update your GitHub Repository Secrets with:"
echo "----------------------------------------------"
echo "AWS_REGION:            $REGION"
echo "AWS_ACCESS_KEY_ID:     <YOUR_ACCESS_KEY>"
echo "AWS_SECRET_ACCESS_KEY: <YOUR_SECRET_KEY>"
echo "AWS_ACCOUNT_ID:        $(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '<RUN aws configure FIRST>')"
echo "ECR_REPOSITORY:        $REPO_NAME"
echo "LAMBDA_ROLE_ARN:       $ROLE_ARN"
echo "MLFLOW_ARTIFACT_ROOT:  s3://$BUCKET_NAME"
echo "----------------------------------------------"
