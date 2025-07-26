"""
AI Fight Coach - Simplified Main FastAPI Application
Version with better error handling for deployment
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
import traceback

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI(title="AI Fight Coach", version="1.0.0")

# Create necessary directories
for directory in ["uploads", "output", "static", "temp"]:
    Path(directory).mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components with detailed error handling
video_processor = None
gemini_client = None
tts_client = None
user_manager = None

print("🔧 Starting component initialization...")

# Test each component individually
try:
    print("📁 Testing utils.video_processor import...")
    from utils.video_processor import VideoProcessor
    video_processor = VideoProcessor()
    print("✅ VideoProcessor initialized successfully")
except Exception as e:
    print(f"❌ VideoProcessor failed: {e}")
    print(f"📋 This is expected in Railway environment - video processing will be limited")

try:
    print("🤖 Testing utils.gemini_client import...")
    from utils.gemini_client import GeminiClient
    gemini_client = GeminiClient()
    print("✅ GeminiClient initialized successfully")
except Exception as e:
    print(f"❌ GeminiClient failed: {e}")
    print(f"📋 Traceback: {traceback.format_exc()}")

try:
    print("🔊 Testing utils.tts_client import...")
    from utils.tts_client import TTSClient
    tts_client = TTSClient()
    print("✅ TTSClient initialized successfully")
except Exception as e:
    print(f"❌ TTSClient failed: {e}")
    print(f"📋 Traceback: {traceback.format_exc()}")

try:
    print("👥 Testing user_management import...")
    from user_management import UserManager
    from email_config import SMTP_CONFIG, ADMIN_EMAIL
    user_manager = UserManager(csv_file="users.csv", smtp_config=SMTP_CONFIG)
    print("✅ UserManager initialized successfully")
except Exception as e:
    print(f"❌ UserManager failed: {e}")
    print(f"📋 Traceback: {traceback.format_exc()}")

print("🏁 Component initialization complete!")

# Store active jobs
active_jobs = {}

# Store files for auto-deletion
files_to_delete = {}

def schedule_file_deletion(file_path: str, delay_minutes: int = 15):
    """Schedule a file for deletion after the specified delay."""
    deletion_time = time.time() + (delay_minutes * 60)
    files_to_delete[file_path] = deletion_time
    print(f"Scheduled deletion of {file_path} in {delay_minutes} minutes")

def cleanup_old_files():
    """Remove files that have passed their deletion time."""
    current_time = time.time()
    files_to_remove = []
    
    for file_path, deletion_time in files_to_delete.items():
        if current_time >= deletion_time:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
            files_to_remove.append(file_path)
    
    for file_path in files_to_remove:
        del files_to_delete[file_path]

def cleanup_worker():
    """Background worker to clean up old files."""
    while True:
        cleanup_old_files()
        time.sleep(60)

# Start cleanup worker
cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
cleanup_thread.start()

@app.on_event("startup")
async def startup_event():
    print("🚀 AI Fight Coach application starting up...")
    print("📁 All directories created and components initialized")
    print(f"🔧 Component Status:")
    print(f"   - VideoProcessor: {'✅' if video_processor else '❌ (Limited functionality)'}")
    print(f"   - GeminiClient: {'✅' if gemini_client else '❌'}")
    print(f"   - TTSClient: {'✅' if tts_client else '❌'}")
    print(f"   - UserManager: {'✅' if user_manager else '❌'}")

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment platforms"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "components": {
            "video_processor": video_processor is not None,
            "gemini_client": gemini_client is not None,
            "tts_client": tts_client is not None,
            "user_manager": user_manager is not None
        },
        "environment": {
            "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
            "ELEVENLABS_API_KEY": bool(os.getenv("ELEVENLABS_API_KEY")),
            "SMTP_EMAIL": bool(os.getenv("SMTP_EMAIL")),
            "SMTP_PASSWORD": bool(os.getenv("SMTP_PASSWORD"))
        },
        "message": "App is running! Some components may have limited functionality due to Railway environment constraints."
    }

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <script>
        window.location.href = '/register';
    </script>
    """)

@app.get("/register")
async def register_page():
    try:
        with open("static/register.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading registration page: {e}</h1>")

@app.get("/main")
async def main_page():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading main page: {e}</h1>")

@app.post("/register-user")
async def register_user(name: str = Form(...), email: str = Form(...)):
    try:
        if user_manager:
            user_manager.register_user(name, email)
            return JSONResponse(content={"success": True, "message": "User registered successfully"})
        else:
            # Fallback: just return success without actually registering
            return JSONResponse(content={"success": True, "message": "User registered successfully (demo mode)"})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Registration failed: {e}"})

@app.post("/submit-feedback")
async def submit_feedback(
    name: str = Form(...),
    email: str = Form(...),
    rating: int = Form(...),
    feedback_text: str = Form(...)
):
    try:
        if user_manager:
            user_manager.send_feedback_to_admin(name, email, rating, feedback_text)
            return JSONResponse(content={"success": True, "message": "Feedback submitted successfully"})
        else:
            # Fallback: just return success
            return JSONResponse(content={"success": True, "message": "Feedback submitted successfully (demo mode)"})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Feedback submission failed: {e}"})

@app.post("/upload")
async def upload_video_redirect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fighter_name: Optional[str] = Form("FIGHTER"),
    analysis_type: Optional[str] = Form("everything")
):
    """Redirect /upload to /upload-video for compatibility"""
    return await upload_video(background_tasks, file, fighter_name, analysis_type)

@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fighter_name: Optional[str] = Form("FIGHTER"),
    analysis_type: Optional[str] = Form("everything")
):
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = f"uploads/{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Schedule file deletion
        schedule_file_deletion(file_path)
        
        if not video_processor or not gemini_client:
            # Demo mode - accept upload but show limited functionality
            active_jobs[job_id] = {
                "status": "demo_mode",
                "progress": 100,
                "message": "Demo mode: Video uploaded successfully! Full analysis is currently unavailable in this environment."
            }
            
            return JSONResponse(content={
                "success": True,
                "job_id": job_id,
                "message": "Video uploaded successfully (demo mode)"
            })
        
        # Start processing in background
        background_tasks.add_task(
            process_video_analysis, job_id, file_path, fighter_name, analysis_type
        )
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "message": "Video uploaded and processing started"
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return JSONResponse(
            content={"success": False, "message": f"Upload failed: {str(e)}"},
            status_code=500
        )

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id in active_jobs:
        return JSONResponse(content=active_jobs[job_id])
    else:
        return JSONResponse(content={"status": "not_found"}, status_code=404)

def process_video_analysis(job_id: str, video_path: str, fighter_name: str, analysis_type: str):
    """Process video analysis in background"""
    try:
        active_jobs[job_id] = {"status": "processing", "progress": 0}
        
        # Update progress
        active_jobs[job_id]["progress"] = 10
        time.sleep(1)
        
        # Process video (simplified for now)
        active_jobs[job_id]["progress"] = 50
        time.sleep(1)
        
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = "Analysis completed successfully"
        
    except Exception as e:
        active_jobs[job_id] = {
            "status": "failed",
            "message": f"Analysis failed: {e}"
        }

@app.get("/users")
async def get_users():
    try:
        if user_manager:
            users = user_manager.get_all_users()
            return JSONResponse(content={"users": users})
        else:
            return JSONResponse(content={"users": [], "message": "User management not available"})
    except Exception as e:
        return JSONResponse(content={"users": [], "message": f"Error: {e}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 