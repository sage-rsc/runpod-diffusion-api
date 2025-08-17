FROM runpod/base:0.4.0-cuda11.8.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create cache directories
RUN mkdir -p /tmp/models /app/cache

# Set default environment variables
ENV MODEL_ID="runwayml/stable-diffusion-v1-5"
ENV MODEL_CACHE_DIR="/tmp/models"
ENV HF_HOME="/app/cache"

# Set the handler
CMD ["python", "-u", "src/handler.py"]
