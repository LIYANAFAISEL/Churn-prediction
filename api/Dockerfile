# Base Python image
FROM python:3.11-slim

# Working directory inside container
WORKDIR /app

# Install dependencies first (better caching)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained model files from root folder
COPY churn_model.pkl /app/churn_model.pkl
COPY scaler.pkl /app/scaler.pkl
COPY feature_columns.pkl /app/feature_columns.pkl

# Copy API code
COPY api/ /app/

# Open port 8000
EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]