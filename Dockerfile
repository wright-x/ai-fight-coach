# Dockerfile for AI Boxing Analysis
FROM python:3.11-slim

# 1. System deps MediaPipe needs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 2. Python deps – use headless OpenCV to cut bloat
RUN pip install --upgrade pip && \
    pip install mediapipe==0.10.21 opencv-python-headless==4.11.0.86 \
               moviepy==1.0.3 fastapi==0.109.2 uvicorn[standard]==0.27.1 \
               python-multipart==0.0.6 google-generativeai==0.3.2 \
               elevenlabs==0.2.27 numpy==1.24.3 Pillow==10.0.0 \
               python-dotenv==1.0.0 aiofiles==23.2.1

# 3. Create app directory
COPY . /app
WORKDIR /app

# 4. Create necessary directories
RUN mkdir -p uploads output static temp

# 5. Expose port
EXPOSE 8080

# 6. Run the app
CMD ["python", "main_simple.py"] 