# Use official Python 3.10 slim image for a smaller footprint
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for compiling some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Expose the port the FastAPI server runs on
EXPOSE 8000

# Set Python path so imports work correctly
ENV PYTHONPATH=/app

# Command to run the application
CMD ["uvicorn", "ml.serving.inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
