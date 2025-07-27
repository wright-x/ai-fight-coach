# Multi-stage Dockerfile for AI Boxing Analysis with Computer Vision
# Based on MediaPipe official Docker guide and community best practices
# This approach properly prepares the system environment before installing Python packages

# Stage 1: Builder stage with all dependencies
FROM python:3.11-slim-bullseye as builder

# Set environment variables for the build stage
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for OpenCV and MediaPipe
# Based on MediaPipe official Docker guide and community solutions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Core system libraries
        curl \
        wget \
        gnupg \
        # OpenCV dependencies (headless)
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        # MediaPipe dependencies
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        # FFmpeg for video processing
        ffmpeg \
        # Additional libraries for stability
        libgtk-3-0 \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libatlas-base-dev \
        gfortran \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --upgrade pip

# Install Python packages with proper dependency resolution
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Lean production stage
FROM python:3.11-slim-bullseye

# Set environment variables for production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Copy only the essential system libraries from builder stage
# This ensures we have the required libraries without the build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Core runtime libraries
        curl \
        # OpenCV runtime dependencies
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        # MediaPipe runtime dependencies
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        # FFmpeg runtime
        ffmpeg \
        # Additional runtime libraries
        libgtk-3-0 \
        libavcodec59 \
        libavformat59 \
        libswscale6 \
        libv4l-0 \
        libxvidcore4 \
        libx264-163 \
        libjpeg62-turbo \
        libpng16-16 \
        libtiff5 \
        libatlas3-base \
        && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory
WORKDIR /app

# Copy application files
COPY . /app

# Create necessary directories
RUN mkdir -p uploads output static temp

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the app
CMD ["python", "main_simple.py"] 