# Use a slim Python 3.12 image as the base
FROM python:3.12-slim

# Set environment variables to optimize Python execution
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for ONNX and building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (~180MB instead of 10GB CUDA binaries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Copy dependency files first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command to boot the LangGraph consumer
CMD ["python", "src/infrastructure/ingestion/consumer.py"]