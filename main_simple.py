"""
AI Fight Coach - Simplified Main FastAPI Application
Version with better error handling for deployment
"""

import os
import uuid
import shutil
import threading
import time
import io
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import json
import logging
import traceback

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# In-memory storage for Railway compatibility
in_memory_files = {}  # {job_id: file_content}
active_jobs = {}

def initialize_components():
    """Initialize all components with detailed logging"""
    global video_processor, gemini_client, tts_client, user_manager
    
    logger.info("🔧 Starting component initialization...")

    # Test each component individually
    try:
        logger.info("📁 Testing utils.video_processor import...")
        from utils.video_processor import VideoProcessor
        video_processor = VideoProcessor()
        logger.info("✅ VideoProcessor initialized successfully")
    except Exception as e:
        logger.error(f"❌ VideoProcessor failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        logger.info("📋 This is expected in Railway environment - video processing will be limited")

    try:
        logger.info("🤖 Testing utils.gemini_client import...")
        from utils.gemini_client import GeminiClient
        gemini_client = GeminiClient()
        logger.info("✅ GeminiClient initialized successfully")
    except Exception as e:
        logger.error(f"❌ GeminiClient failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        logger.info("📋 This is expected in Railway environment - AI analysis will be limited")

    try:
        logger.info("🔊 Testing utils.tts_client import...")
        from utils.tts_client import TTSClient
        tts_client = TTSClient()
        logger.info("✅ TTSClient initialized successfully")
    except Exception as e:
        logger.error(f"❌ TTSClient failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        logger.info("📋 This is expected in Railway environment - TTS will be limited")

    try:
        logger.info("👥 Testing user_management import...")
        from user_management import UserManager
        from email_config import SMTP_CONFIG, ADMIN_EMAIL
        user_manager = UserManager(csv_file="users.csv", smtp_config=SMTP_CONFIG)
        logger.info("✅ UserManager initialized successfully")
    except Exception as e:
        logger.error(f"❌ UserManager failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")

    logger.info("🏁 Component initialization complete!")
    logger.info(f"📊 Component Status:")
    logger.info(f"   - VideoProcessor: {'✅' if video_processor else '❌'}")
    logger.info(f"   - GeminiClient: {'✅' if gemini_client else '❌'}")
    logger.info(f"   - TTSClient: {'✅' if tts_client else '❌'}")
    logger.info(f"   - UserManager: {'✅' if user_manager else '❌'}")

def schedule_file_deletion(job_id: str, delay_minutes: int = 15):
    """Schedule in-memory file deletion after the specified delay."""
    def delete_file():
        time.sleep(delay_minutes * 60)
        if job_id in in_memory_files:
            del in_memory_files[job_id]
            logger.info(f"🗑️ Deleted in-memory file for job {job_id}")
    
    thread = threading.Thread(target=delete_file, daemon=True)
    thread.start()

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("🚀 AI Fight Coach starting up...")
    logger.info("📁 Creating directories...")
    
    # Create directories
    for directory in ["uploads", "output", "static", "temp"]:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

    initialize_components()

@app.get("/health")
async def health_check():
    logger.info("🏥 Health check requested")
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
    """Redirect to main page"""
    logger.info("🏠 Root page requested")
    return HTMLResponse(content="""
    <html>
        <head><title>AI Fight Coach</title></head>
        <body>
            <h1>AI Fight Coach</h1>
            <p><a href="/main">Go to Main Application</a></p>
            <p><a href="/register">Register</a></p>
        </body>
    </html>
    """)

@app.get("/register")
async def register_page():
    """Serve registration page"""
    logger.info("📝 Registration page requested")
    try:
        with open("static/register.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("❌ Registration page not found")
        return HTMLResponse(content="<h1>Registration page not found</h1>")

@app.get("/main")
async def main_page():
    """Serve main application page"""
    logger.info("🎯 Main page requested")
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("❌ Main page not found")
        return HTMLResponse(content="<h1>Main page not found</h1>")

@app.post("/register-user")
async def register_user(name: str = Form(...), email: str = Form(...)):
    """Register a new user"""
    logger.info(f"👤 User registration: {name} ({email})")
    try:
        if user_manager:
            user_manager.register_user(name, email)
            return JSONResponse(content={"success": True, "message": "Registration successful!"})
        else:
            return JSONResponse(content={"success": True, "message": "Registration successful! (demo mode)"})
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
        return JSONResponse(content={"success": False, "message": f"Registration failed: {e}"})

@app.post("/submit-feedback")
async def submit_feedback(
    name: str = Form(...),
    email: str = Form(...),
    rating: int = Form(...),
    feedback_text: str = Form(...)
):
    """Submit user feedback"""
    logger.info(f"📝 Feedback submitted: {name} ({email}) - Rating: {rating}")
    try:
        if user_manager:
            user_manager.send_feedback_email(name, email, rating, feedback_text)
            return JSONResponse(content={"success": True, "message": "Feedback submitted successfully!"})
        else:
            return JSONResponse(content={"success": True, "message": "Feedback submitted successfully! (demo mode)"})
    except Exception as e:
        logger.error(f"❌ Feedback submission failed: {e}")
        return JSONResponse(content={"success": False, "message": f"Feedback submission failed: {e}"})

@app.post("/upload")
async def upload_video_redirect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fighter_name: Optional[str] = Form("FIGHTER"),
    analysis_type: Optional[str] = Form("everything")
):
    """Redirect /upload to /upload-video for compatibility"""
    logger.info("🔄 Upload redirect requested")
    return await upload_video(background_tasks, file, fighter_name, analysis_type)

@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fighter_name: Optional[str] = Form("FIGHTER"),
    analysis_type: Optional[str] = Form("everything")
):
    logger.info(f"📤 Video upload started: {file.filename} ({file.content_type})")
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        logger.info(f"🆔 Generated job ID: {job_id}")
        
        # Read file content into memory (Railway-compatible)
        file_content = await file.read()
        in_memory_files[job_id] = {
            "content": file_content,
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        # Schedule in-memory file deletion
        schedule_file_deletion(job_id, 15)
        
        logger.info(f"✅ Video uploaded to memory: {file.filename} ({len(file_content)} bytes)")
        
        # Check if components are available
        logger.info(f"🔍 Checking component availability:")
        logger.info(f"   - VideoProcessor: {'✅' if video_processor else '❌'}")
        logger.info(f"   - GeminiClient: {'✅' if gemini_client else '❌'}")
        
        if not video_processor or not gemini_client:
            # Demo mode - accept upload but show limited functionality
            logger.info(f"🎭 Demo mode activated for job {job_id}")
            active_jobs[job_id] = {
                "status": "demo_mode",
                "progress": 100,
                "message": "Demo mode: Video uploaded successfully! Full analysis is currently unavailable in this environment.",
                "video_url": f"/video/{job_id}",  # Use our custom endpoint
                "analysis_result": {
                    "highlights": [
                        {
                            "timestamp": 15,
                            "detailed_feedback": "Demo mode: This is a sample highlight",
                            "action_required": "Demo mode: Sample action required"
                        }
                    ],
                    "recommended_drills": [
                        {
                            "drill_name": "Demo Drill",
                            "description": "This is a sample drill for demonstration purposes",
                            "problem_it_fixes": "Demo mode: Sample problem fix"
                        }
                    ]
                }
            }
            
            return JSONResponse(content={
                "success": True,
                "job_id": job_id,
                "message": "Video uploaded successfully (demo mode)"
            })
        
        # Start processing in background
        logger.info(f"🚀 Starting background processing for job {job_id}")
        background_tasks.add_task(
            process_video_analysis, job_id, fighter_name, analysis_type
        )
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "message": "Video uploaded and processing started"
        })
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return JSONResponse(
            content={"success": False, "message": f"Upload failed: {str(e)}"},
            status_code=500
        )

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    logger.info(f"📊 Status check requested for job: {job_id}")
    if job_id in active_jobs:
        status = active_jobs[job_id]
        logger.info(f"📈 Job {job_id} status: {status.get('status', 'unknown')}")
        return JSONResponse(content=status)
    else:
        logger.warning(f"⚠️ Job {job_id} not found")
        return JSONResponse(content={"status": "not_found"}, status_code=404)

@app.get("/video/{job_id}")
async def serve_video(job_id: str):
    """Serve the uploaded video"""
    logger.info(f"🎥 Video request for job: {job_id}")
    try:
        if job_id in in_memory_files:
            file_info = in_memory_files[job_id]
            logger.info(f"✅ Serving video: {file_info['filename']} ({len(file_info['content'])} bytes)")
            return Response(
                content=file_info["content"],
                media_type=file_info["content_type"],
                headers={"Content-Disposition": f"inline; filename={file_info['filename']}"}
            )
        else:
            logger.error(f"❌ Video not found for job: {job_id}")
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
    except Exception as e:
        logger.error(f"❌ Error serving video for job {job_id}: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return JSONResponse(content={"error": f"Error serving video: {e}"}, status_code=500)

def process_video_analysis(job_id: str, fighter_name: str, analysis_type: str):
    """Process video analysis in background"""
    logger.info(f"🔍 Starting video analysis for job: {job_id}")
    try:
        active_jobs[job_id] = {"status": "processing", "progress": 0}
        
        # Check if file exists in memory
        if job_id not in in_memory_files:
            logger.error(f"❌ Video file not found in memory for job: {job_id}")
            active_jobs[job_id] = {
                "status": "failed",
                "message": "Video file not found in memory"
            }
            return
        
        # Update progress
        active_jobs[job_id]["progress"] = 10
        logger.info(f"📈 Job {job_id} progress: 10%")
        
        # Save video file temporarily for Gemini analysis
        file_info = in_memory_files[job_id]
        temp_video_path = f"temp/video_{job_id}.mp4"
        
        with open(temp_video_path, 'wb') as f:
            f.write(file_info["content"])
        
        logger.info(f"💾 Saved video to temp file: {temp_video_path}")
        
        # Update progress
        active_jobs[job_id]["progress"] = 30
        logger.info(f"📈 Job {job_id} progress: 30%")
        
        # Analyze with Gemini
        if gemini_client:
            logger.info(f"🤖 Calling Gemini analysis for job: {job_id}")
            analysis_result = gemini_client.analyze_video(temp_video_path, analysis_type)
            logger.info(f"✅ Gemini analysis completed for job: {job_id}")
        else:
            logger.warning(f"⚠️ Gemini client not available, using mock analysis")
            analysis_result = {
                "highlights": [
                    {
                        "timestamp": 15,
                        "detailed_feedback": "Mock analysis: Good technique observed",
                        "action_required": "Continue practicing"
                    }
                ],
                "recommended_drills": [
                    {
                        "drill_name": "Mock Drill",
                        "description": "This is a mock drill for demonstration",
                        "problem_it_fixes": "Mock problem fix"
                    }
                ]
            }
        
        # Update progress
        active_jobs[job_id]["progress"] = 80
        logger.info(f"📈 Job {job_id} progress: 80%")
        
        # Clean up temp file
        try:
            os.remove(temp_video_path)
            logger.info(f"🗑️ Cleaned up temp file: {temp_video_path}")
        except:
            pass
        
        # Complete processing
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = "Analysis completed successfully"
        active_jobs[job_id]["video_url"] = f"/video/{job_id}"  # Use our custom endpoint
        active_jobs[job_id]["analysis_result"] = analysis_result
        
        logger.info(f"✅ Analysis completed for job: {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for job {job_id}: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        active_jobs[job_id] = {
            "status": "failed",
            "message": f"Analysis failed: {e}"
        }

@app.get("/users")
async def get_users():
    logger.info("👥 Users list requested")
    try:
        if user_manager:
            users = user_manager.get_all_users()
            return JSONResponse(content={"users": users})
        else:
            return JSONResponse(content={"users": [], "message": "User management not available"})
    except Exception as e:
        logger.error(f"❌ Error getting users: {e}")
        return JSONResponse(content={"users": [], "message": f"Error: {e}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 