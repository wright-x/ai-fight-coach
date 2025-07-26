"""
AI Fight Coach - Main FastAPI Application
Complete video analysis pipeline with Gemini AI and ElevenLabs TTS.
"""

import os
import uuid
import shutil
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import json
import logging

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.video_processor import VideoProcessor
from utils.gemini_client import GeminiClient
from utils.tts_client import TTSClient
from utils.logger import logger
from user_management import UserManager
from email_config import SMTP_CONFIG, ADMIN_EMAIL

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
video_processor = VideoProcessor()
gemini_client = GeminiClient()
tts_client = TTSClient()

# Initialize user manager with SMTP configuration
user_manager = UserManager(csv_file="users.csv", smtp_config=SMTP_CONFIG)

# Create necessary directories
for directory in ["uploads", "output", "static", "temp"]:
    Path(directory).mkdir(exist_ok=True)

# Store active jobs
active_jobs = {}

# Store files for auto-deletion
files_to_delete = {}  # {file_path: deletion_time}

def schedule_file_deletion(file_path: str, delay_minutes: int = 15):
    """Schedule a file for deletion after the specified delay."""
    deletion_time = time.time() + (delay_minutes * 60)
    files_to_delete[file_path] = deletion_time
    logger.info(f"Scheduled deletion of {file_path} in {delay_minutes} minutes")

def cleanup_old_files():
    """Remove files that have passed their deletion time."""
    current_time = time.time()
    files_to_remove = []
    
    for file_path, deletion_time in files_to_delete.items():
        if current_time >= deletion_time:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted old file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
            files_to_remove.append(file_path)
    
    for file_path in files_to_remove:
        del files_to_delete[file_path]

def cleanup_worker():
    """Background worker to clean up old files."""
    while True:
        cleanup_old_files()
        time.sleep(60)  # Check every minute

# Start cleanup worker
cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
cleanup_thread.start()

@app.on_event("startup")
async def startup_event():
    logger.info("AI Fight Coach application starting up...")
    logger.info("All directories created and components initialized")

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment platforms"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <script>
        window.location.href = '/register';
    </script>
    """)

@app.get("/register")
async def register_page():
    with open("static/register.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/main")
async def main_page():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/register-user")
async def register_user(name: str = Form(...), email: str = Form(...)):
    """Register a new user"""
    try:
        success = user_manager.register_user(name, email)
        if success:
            return JSONResponse(content={"success": True, "message": "Registration successful!"})
        else:
            return JSONResponse(content={"success": False, "message": "Registration failed. Please try again."})
    except Exception as e:
        logger.error(f"Error in user registration: {e}")
        return JSONResponse(content={"success": False, "message": "Registration failed. Please try again."})

@app.post("/submit-feedback")
async def submit_feedback(
    name: str = Form(...),
    email: str = Form(...),
    rating: int = Form(...),
    feedback_text: str = Form(...)
):
    """Submit user feedback"""
    try:
        user_manager.send_feedback_to_admin(name, email, rating, feedback_text)
        return JSONResponse(content={"success": True, "message": "Feedback submitted successfully!"})
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return JSONResponse(content={"success": False, "message": "Failed to submit feedback. Please try again."})

@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fighter_name: Optional[str] = Form("FIGHTER"),
    analysis_type: Optional[str] = Form("everything")
):
    """Upload and analyze a boxing video"""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        file_path = f"uploads/{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Schedule file for deletion
        schedule_file_deletion(file_path, 15)
        
        # Store job info
        active_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "fighter_name": fighter_name,
            "analysis_type": analysis_type
        }
        
        # Start background processing
        background_tasks.add_task(
            process_video_analysis, 
            job_id, 
            file_path, 
            fighter_name, 
            analysis_type
        )
        
        logger.info(f"Started video analysis for job {job_id}")
        return {"job_id": job_id, "status": "processing"}
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        return {"error": "Upload failed"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a video analysis job"""
    if job_id not in active_jobs:
        return {"error": "Job not found"}
    
    job_info = active_jobs[job_id]
    return job_info

def process_video_analysis(job_id: str, video_path: str, fighter_name: str, analysis_type: str):
    """Process video analysis in background"""
    try:
        logger.info(f"Processing video analysis for job {job_id}")
        
        # Update progress
        active_jobs[job_id]["progress"] = 10
        active_jobs[job_id]["status"] = "analyzing"
        
        # Map analysis type to prompt file
        prompt_file_map = {
            "everything": "prompts/everything_prompt.txt",
            "head_movement": "prompts/head_movement_prompt.txt",
            "punch_techniques": "prompts/punch_techniques_prompt.txt",
            "footwork": "prompts/footwork_prompt.txt"
        }
        
        prompt_file = prompt_file_map.get(analysis_type, "prompts/everything_prompt.txt")
        
        # Update progress
        active_jobs[job_id]["progress"] = 30
        time.sleep(1)
        
        # Analyze video with Gemini
        analysis_result = gemini_client.analyze_video(video_path, prompt_file)
        
        # Update progress
        active_jobs[job_id]["progress"] = 60
        active_jobs[job_id]["status"] = "creating_video"
        time.sleep(1)
        
        # Create highlight video
        highlight_video_path = f"static/final_{job_id}.mp4"
        video_processor.create_highlight_video(
            video_path, 
            analysis_result.get('highlights', []), 
            highlight_video_path,
            fighter_name
        )
        
        # Update progress
        active_jobs[job_id]["progress"] = 90
        time.sleep(1)
        
        # Ensure the video file is accessible
        try:
            # Test file access
            with open(highlight_video_path, 'rb') as f:
                f.read(1024)  # Read first 1KB to test access
            logger.info(f"Video file is accessible: {highlight_video_path}")
        except Exception as e:
            logger.error(f"Video file access error: {e}")
            raise ValueError(f"Video file is not accessible: {highlight_video_path}")
        
        # Update job status
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["result"] = {
            "video_url": f"/static/final_{job_id}.mp4",
            "analysis_result": analysis_result
        }
        
        # Schedule final video for deletion
        schedule_file_deletion(highlight_video_path, 15)
        
        # Send analysis results email if user is registered
        try:
            # Get user info from localStorage (this would need to be passed from frontend)
            # For now, we'll try to send to a default email or skip
            user_email = "user@example.com"  # This should come from the frontend
            user_name = fighter_name
            video_url = f"http://localhost:8000/static/final_{job_id}.mp4"
            
            # Only send email if we have a valid user email
            if user_email and user_email != "user@example.com":
                user_manager.send_analysis_results_email(
                    user_name, 
                    user_email, 
                    analysis_result, 
                    video_url
                )
        except Exception as e:
            logger.error(f"Error sending analysis results email: {e}")
        
        logger.info(f"Video analysis completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in video analysis: {e}")
        active_jobs[job_id]["status"] = "error"
        active_jobs[job_id]["error"] = str(e)

@app.get("/users")
async def get_users():
    """Get all registered users (admin endpoint)"""
    try:
        users = user_manager.get_all_users()
        return JSONResponse(content={"users": users})
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return JSONResponse(content={"error": "Failed to get users"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 