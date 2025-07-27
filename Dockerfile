# Minimal Dockerfile for AI Boxing Analysis
FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install only essential system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.109.2 \
        uvicorn[standard]==0.27.1 \
        python-multipart==0.0.6 \
        google-generativeai==0.3.2 \
        elevenlabs==0.2.27 \
        numpy==1.24.3 \
        Pillow==10.0.0 \
        python-dotenv==1.0.0 \
        aiofiles==23.2.1 \
        itsdangerous==2.1.2

# Install OpenCV and MediaPipe separately to avoid conflicts
RUN pip install --no-cache-dir opencv-python-headless==4.11.0.86
RUN pip install --no-cache-dir mediapipe==0.10.21

# Install MoviePy last
RUN pip install --no-cache-dir moviepy==1.0.3

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