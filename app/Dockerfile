# AWS Lambda Python Base Image
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies (needed for some python packages or xgboost)
# libgomp1 is often needed for xgboost/sklearn concurrency
RUN yum update -y && \
    yum install -y libgomp && \
    yum clean all

# Copy requirements into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models to LAMBDA_TASK_ROOT
COPY main.py ${LAMBDA_TASK_ROOT}
COPY model_xgb.joblib ${LAMBDA_TASK_ROOT}
COPY model_iso.joblib ${LAMBDA_TASK_ROOT}
COPY cashier_stats.joblib ${LAMBDA_TASK_ROOT}

# Set the CMD to the handler (Mangum adapter)
CMD [ "main.handler" ]
