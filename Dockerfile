# Base image — slim variant reduces image size significantly
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by pypdf and sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first — Docker caches this layer
# separately from the application code, so rebuilds are fast when only
# code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories that must exist at runtime
RUN mkdir -p data/arxiv data/uploads vectorstore/chroma

# Expose the FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]