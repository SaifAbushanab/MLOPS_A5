# Dockerfile - Mock deployment container for the trained model

FROM python:3.10-slim

# Accept the MLflow Run ID as a build argument
ARG RUN_ID

# Simulate downloading the model from MLflow
RUN echo "Downloading model for Run ID: ${RUN_ID}"

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "-c", "print('Model server is running...')"]
