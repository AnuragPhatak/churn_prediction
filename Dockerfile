FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for XGBoost, scikit-learn, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-train.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy trained MLflow model (local pyfunc model)
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
