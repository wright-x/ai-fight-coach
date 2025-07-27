# Ultra-minimal Dockerfile for AI Boxing Analysis
FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install only the absolute minimum system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install Python packages one by one to avoid conflicts
RUN pip install --upgrade pip

# Core web framework
RUN pip install --no-cache-dir fastapi==0.109.2
RUN pip install --no-cache-dir uvicorn[standard]==0.27.1
RUN pip install --no-cache-dir python-multipart==0.0.6

# AI and API services
RUN pip install --no-cache-dir google-generativeai==0.3.2
RUN pip install --no-cache-dir elevenlabs==0.2.27

# Basic utilities
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir Pillow==10.0.0
RUN pip install --no-cache-dir python-dotenv==1.0.0
RUN pip install --no-cache-dir aiofiles==23.2.1
RUN pip install --no-cache-dir itsdangerous==2.1.2

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