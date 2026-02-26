# Use a stable PyTorch image with necessary CUDA and cuDNN libraries.
# Using a specific version (e.g., 2.1.0 with CUDA 11.8) ensures compatibility.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build tools
RUN apt-get update && apt-get install -y \
    tzdata \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
# This includes all scripts (train.py, infer.py, etc.) and project folders.

# Download the data
# Set the default entrypoint to python
CMD ["python", "train.py"]
