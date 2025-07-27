# Use the official Python 3.11 slim image based on Debian Bullseye
FROM python:3.11-slim-bullseye

# Set environment variables to prevent interactive prompts and ensure Python output is unbuffered
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Update apt and install the CORRECT system dependencies for Debian Bullseye.
# This list is minimal and targets the build requirements for our Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools
    build-essential \
    cmake \
    # Core OpenCV & MediaPipe system libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # FFmpeg development libraries - THIS IS THE KEY FIX
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    # GStreamer libraries for MediaPipe
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    # Additional libraries for stability
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Upgrade pip and install Python packages from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads output static temp

# Expose port dynamically (Railway will set $PORT)
EXPOSE $PORT

# Health check using dynamic port
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run the application (Python code handles port binding)
CMD ["python", "main_simple.py"] 