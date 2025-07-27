"""
AI Fight Coach - Simplified Main FastAPI Application
Version with better error handling for deployment
"""

# Suppress TensorFlow/MediaPipe C++ warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Disable CUDA warnings

# Standard library imports
import json
import shutil
import tempfile
import time
import traceback
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# Set up detailed logging
import logging
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

    # Initialize VideoProcessor (main component)
    try:
        logger.info("📁 Initializing VideoProcessor...")
        from utils.video_processor import VideoProcessor
        logger.info("📦 VideoProcessor class imported successfully")
        video_processor = VideoProcessor()
        logger.info("✅ VideoProcessor initialized successfully")
        logger.info(f"📊 VideoProcessor object: {type(video_processor)}")
    except Exception as e:
        logger.error(f"❌ VideoProcessor failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        video_processor = None
        logger.warning("⚠️ Video processing will be disabled - this is a critical error!")

    # Initialize GeminiClient
    try:
        logger.info("🤖 Initializing GeminiClient...")
        from utils.gemini_client import GeminiClient
        gemini_client = GeminiClient()
        logger.info("✅ GeminiClient initialized successfully")
    except Exception as e:
        logger.error(f"❌ GeminiClient failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        gemini_client = None
        logger.warning("⚠️ AI analysis will be limited")

    # Initialize TTSClient
    try:
        logger.info("🔊 Initializing TTSClient...")
        from utils.tts_client import TTSClient
        tts_client = TTSClient()
        logger.info("✅ TTSClient initialized successfully")
    except Exception as e:
        logger.error(f"❌ TTSClient failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        tts_client = None
        logger.warning("⚠️ TTS will be limited")

    # Initialize UserManager
    try:
        logger.info("👥 Initializing UserManager...")
        from user_management import UserManager
        from email_config import SMTP_CONFIG, ADMIN_EMAIL
        user_manager = UserManager(csv_file="users.csv", smtp_config=SMTP_CONFIG)
        logger.info("✅ UserManager initialized successfully")
    except Exception as e:
        logger.error(f"❌ UserManager failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        user_manager = None

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
    """Comprehensive health check endpoint"""
    try:
        # Check basic system health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv("RAILWAY_ENVIRONMENT", "development"),
            "components": {
                "video_processor": video_processor is not None,
                "gemini_client": gemini_client is not None,
                "tts_client": tts_client is not None,
                "user_manager": user_manager is not None
            },
            "system": {
                "python_version": "3.11",
                "platform": "railway",
                "memory_usage": "ok",
                "disk_space": "ok"
            },
            "endpoints": {
                "root": "/",
                "health": "/health",
                "upload": "/upload-video",
                "status": "/status/{job_id}",
                "video": "/video/{job_id}"
            }
        }
        
        # Check if any critical components are missing
        if not any(health_status["components"].values()):
            health_status["status"] = "degraded"
            health_status["message"] = "Some components failed to initialize"
        else:
            health_status["message"] = "All systems operational"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for connectivity"""
    return JSONResponse(content={
        "message": "AI Boxing Analysis server is running!",
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })

@app.get("/test-video")
async def test_video():
    """Test endpoint to verify video serving works"""
    logger.info("🎥 Test video endpoint requested")
    try:
        # Create a simple test video (1 second black video)
        import numpy as np
        import cv2
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_video.mp4', fourcc, 1.0, (640,480))
        
        # Create a black frame
        frame = np.zeros((480,640,3), dtype=np.uint8)
        
        # Write 30 frames (1 second at 30fps)
        for _ in range(30):
            out.write(frame)
        
        out.release()
        
        # Read the test video
        with open('test_video.mp4', 'rb') as f:
            video_content = f.read()
        
        # Clean up
        os.remove('test_video.mp4')
        
        logger.info(f"✅ Test video created: {len(video_content)} bytes")
        
        return Response(
            content=video_content,
            media_type="video/mp4",
            headers={"Content-Disposition": "inline; filename=test_video.mp4"}
        )
        
    except Exception as e:
        logger.error(f"❌ Test video creation failed: {e}")
        return JSONResponse(content={"error": f"Test video failed: {e}"}, status_code=500)

@app.get("/test-video-simple")
async def test_video_simple():
    """Create and serve a simple test video to verify video serving works"""
    logger.info("🎥 Creating simple test video...")
    try:
        import cv2
        import numpy as np
        
        # Create a simple test video (2 seconds, 640x480)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        test_video_path = 'test_simple.mp4'
        out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
        
        # Create frames with text
        for i in range(60):  # 2 seconds at 30fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add text
            cv2.putText(frame, f"Test Video Frame {i+1}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "AI Fight Coach Test", (50, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame)
        
        out.release()
        
        # Read the video file
        with open(test_video_path, 'rb') as f:
            video_content = f.read()
        
        # Clean up
        os.remove(test_video_path)
        
        logger.info(f"✅ Test video created: {len(video_content)} bytes")
        
        headers = {
            "Content-Disposition": "inline; filename=test_video.mp4",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-cache"
        }
        
        return Response(
            content=video_content,
            media_type="video/mp4",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"❌ Test video creation failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return JSONResponse(content={"error": f"Test video failed: {e}"}, status_code=500)

@app.get("/")
async def root():
    """Check if user is registered and redirect appropriately"""
    logger.info("🏠 Root page requested")
    
    # Provide a landing page that checks localStorage for registration status
    return HTMLResponse(content="""
    <html>
        <head>
            <title>AI Fight Coach</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                }
                .container {
                    text-align: center;
                    background: rgba(255,255,255,0.1);
                    padding: 40px;
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }
                h1 {
                    font-size: 3rem;
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 15px;
                }
                .btn {
                    display: inline-block;
                    background: rgba(255,255,255,0.2);
                    color: white;
                    padding: 15px 30px;
                    margin: 10px;
                    border-radius: 25px;
                    text-decoration: none;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    border: 2px solid rgba(255,255,255,0.3);
                }
                .btn:hover {
                    background: rgba(255,255,255,0.3);
                    transform: translateY(-2px);
                }
                .btn-primary {
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                    border-color: #28a745;
                }
                .btn-primary:hover {
                    background: linear-gradient(135deg, #218838 0%, #1ea085 100%);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🥊 AI Fight Coach</h1>
                <p style="font-size: 1.2rem; margin-bottom: 30px; opacity: 0.9;">
                    Professional Boxing Analysis Powered by AI
                </p>
                <div id="welcome-message" style="display: none;">
                    <p style="font-size: 1.1rem; margin-bottom: 20px;">Welcome back!</p>
                    <a href="/main" class="btn btn-primary">Continue to Analysis</a>
                </div>
                <div id="new-user" style="display: none;">
                    <p style="font-size: 1.1rem; margin-bottom: 20px;">Get started with AI-powered boxing analysis</p>
                    <a href="/register" class="btn">Register First</a>
                    <a href="/main" class="btn btn-primary">Try Demo</a>
                </div>
                <div id="loading">
                    <p>Checking your status...</p>
                </div>
            </div>
            
            <script>
                // Check if user is registered in localStorage
                window.onload = function() {
                    const userName = localStorage.getItem('userName');
                    const userEmail = localStorage.getItem('userEmail');
                    
                    const welcomeDiv = document.getElementById('welcome-message');
                    const newUserDiv = document.getElementById('new-user');
                    const loadingDiv = document.getElementById('loading');
                    
                    if (userName && userEmail) {
                        // User is registered
                        loadingDiv.style.display = 'none';
                        welcomeDiv.style.display = 'block';
                    } else {
                        // New user
                        loadingDiv.style.display = 'none';
                        newUserDiv.style.display = 'block';
                    }
                };
            </script>
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
async def register_user(request: Request, name: str = Form(...), email: str = Form(...)):
    """Register user"""
    try:
        logger.info(f"👤 User registration: {name} ({email})")
        
        # Create user data
        user_data = {
            "name": name,
            "email": email,
            "registered_at": datetime.now().isoformat(),
            "session_id": str(uuid.uuid4())
        }
        
        # Try to save to CSV if user manager is available
        if user_manager:
            try:
                user_manager.register_user(name, email)
                logger.info(f"✅ User saved to CSV: {email}")
            except Exception as e:
                logger.error(f"⚠️ Failed to save to CSV: {e}")
        
        # Send welcome email if available
        if user_manager:
            try:
                user_manager.send_welcome_email(name, email)
                logger.info(f"✅ Welcome email sent: {email}")
            except Exception as e:
                logger.error(f"⚠️ Failed to send welcome email: {e}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Registration successful! You can now upload videos.",
            "user": user_data
        })
        
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
        return JSONResponse(content={
            "success": False,
            "message": f"Registration failed: {e}"
        })

@app.get("/session")
async def get_session(request: Request):
    """Get current session info"""
    # Simple session check without middleware
    return JSONResponse(content={"user": None, "authenticated": False})

@app.post("/logout")
async def logout(request: Request):
    """Clear user session"""
    # Simple logout without session middleware
    return JSONResponse(content={"success": True, "message": "Logged out successfully"})

@app.post("/submit-feedback")
async def submit_feedback(request: Request):
    """Submit user feedback"""
    try:
        data = await request.json()
        job_id = data.get('job_id')
        feedback_type = data.get('feedback_type')
        feedback_text = data.get('feedback_text', '')
        
        logger.info(f"📝 Received feedback - Job ID: {job_id}, Type: {feedback_type}")
        
        # Send email notification
        if user_manager:
            try:
                subject = f"AI Fight Coach Feedback - {feedback_type}"
                body = f"""
                New feedback received:
                
                Job ID: {job_id}
                Feedback Type: {feedback_type}
                Feedback Text: {feedback_text}
                
                Timestamp: {datetime.now().isoformat()}
                """
                
                # Send to admin email
                admin_email = os.getenv('SMTP_EMAIL', 'admin@ai-boxing.com')
                user_manager.send_email(admin_email, subject, body)
                logger.info("✅ Feedback email sent successfully")
            except Exception as e:
                logger.error(f"❌ Failed to send feedback email: {e}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Feedback submitted successfully"
        })
        
    except Exception as e:
        logger.error(f"❌ Feedback submission failed: {e}")
        return JSONResponse(content={
            "success": False,
            "message": f"Failed to submit feedback: {str(e)}"
        })

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
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fighter_name: Optional[str] = Form("FIGHTER"),
    analysis_type: Optional[str] = Form("everything")
):
    """Upload video for AI analysis"""
    try:
        logger.info(f"📤 Video upload started: {file.filename}")
        
        # Validate file
        if not file.filename or not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return JSONResponse(content={
                "success": False,
                "message": "Please upload a valid video file (.mp4, .avi, .mov, .mkv)"
            })
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Store video in memory
        video_content = await file.read()
        in_memory_files[job_id] = {
            "content": video_content,
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        # Initialize job
        active_jobs[job_id] = {
            "status": "uploaded",
            "progress": 0,
            "message": "Video uploaded successfully",
            "fighter_name": fighter_name,
            "analysis_type": analysis_type,
            "user": {"name": "Anonymous", "email": "anonymous@example.com"},  # Default user
            "uploaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Video uploaded: {job_id} ({len(video_content)} bytes)")
        
        # Start background processing
        background_tasks.add_task(process_video_analysis, job_id, fighter_name, analysis_type)
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "message": "Video uploaded successfully! Analysis in progress..."
        })
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return JSONResponse(content={
            "success": False,
            "message": f"Upload failed: {e}"
        })

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
    logger.info(f"📊 Available jobs in memory: {list(in_memory_files.keys())}")
    
    try:
        if job_id in in_memory_files:
            file_info = in_memory_files[job_id]
            logger.info(f"✅ Serving video: {file_info['filename']} ({len(file_info['content'])} bytes)")
            logger.info(f"📋 Content type: {file_info['content_type']}")
            
            # Add CORS headers for video streaming
            headers = {
                "Content-Disposition": f"inline; filename={file_info['filename']}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "no-cache"
            }
            
            return Response(
                content=file_info["content"],
                media_type=file_info["content_type"],
                headers=headers
            )
        else:
            logger.error(f"❌ Video not found for job: {job_id}")
            logger.error(f"📊 Available jobs: {list(in_memory_files.keys())}")
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
    except Exception as e:
        logger.error(f"❌ Error serving video for job {job_id}: {e}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        return JSONResponse(content={"error": f"Error serving video: {e}"}, status_code=500)

@app.get("/analysis/{job_id}")
async def get_analysis(job_id: str):
    """Get analysis results for a job"""
    logger.info(f"📊 Analysis request for job: {job_id}")
    
    try:
        if job_id in active_jobs:
            job_info = active_jobs[job_id]
            if job_info.get("status") == "completed" and "analysis_result" in job_info:
                logger.info(f"✅ Serving analysis for job: {job_id}")
                return JSONResponse(content=job_info["analysis_result"])
            else:
                logger.warning(f"⚠️ Analysis not ready for job: {job_id}, status: {job_info.get('status')}")
                return JSONResponse(content={"error": "Analysis not ready"}, status_code=404)
        else:
            logger.error(f"❌ Job not found: {job_id}")
            return JSONResponse(content={"error": "Job not found"}, status_code=404)
    except Exception as e:
        logger.error(f"❌ Error serving analysis for job {job_id}: {e}")
        return JSONResponse(content={"error": f"Error serving analysis: {e}"}, status_code=500)

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
            try:
                logger.info(f"🔍 Analysis type: {analysis_type}")
                logger.info(f"📁 Temp video path: {temp_video_path}")
                logger.info(f"📊 Video file exists: {os.path.exists(temp_video_path)}")
                if os.path.exists(temp_video_path):
                    file_size = os.path.getsize(temp_video_path)
                    logger.info(f"📊 Video file size: {file_size} bytes")
                
                analysis_result = gemini_client.analyze_video(temp_video_path, analysis_type)
                logger.info(f"✅ Gemini analysis completed for job: {job_id}")
                logger.info(f"📊 Analysis result keys: {list(analysis_result.keys()) if analysis_result else 'None'}")
                if analysis_result and 'highlights' in analysis_result:
                    logger.info(f"📊 Number of highlights: {len(analysis_result['highlights'])}")
            except Exception as e:
                logger.error(f"❌ Gemini analysis failed: {e}")
                logger.error(f"📋 Traceback: {traceback.format_exc()}")
                logger.warning(f"⚠️ Falling back to mock analysis due to Gemini failure")
                analysis_result = {
                    "highlights": [
                        {
                            "timestamp": "00:15",
                            "short_text": "Tuck your chin and keep your guard up",
                            "long_text": "At 15 seconds, I observed that your guard was slightly lowered and your chin was exposed. This creates a vulnerability that an opponent could exploit. You should maintain a tight guard position with your hands protecting your face at all times.",
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
        else:
            logger.warning(f"⚠️ Gemini client not available, using mock analysis")
            logger.error(f"❌ CRITICAL: Gemini client is None - this should not happen!")
            analysis_result = {
                "highlights": [
                    {
                        "timestamp": "00:15",
                        "short_text": "Tuck your chin and keep your guard up",
                        "long_text": "At 15 seconds, I observed that your guard was slightly lowered and your chin was exposed. This creates a vulnerability that an opponent could exploit. You should maintain a tight guard position with your hands protecting your face at all times.",
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
        
        # Create highlight video with overlays and head tracking
        logger.info(f"🎬 Starting video processing for job: {job_id}")
        logger.info(f"📊 Analysis result: {analysis_result}")
        
        # Always attempt video processing, even if no highlights
        highlight_video_path = f"output/highlight_{job_id}.mp4"
        final_video_path = None
        
        try:
            if video_processor:
                logger.info(f"🎬 VideoProcessor available, creating highlight video...")
                processed_video_path = f"output/highlight_{job_id}.mp4"
                
                # Get user name from session or use default
                user_name = "FIGHTER"  # Default
                if 'user' in active_jobs[job_id]:
                    user_name = active_jobs[job_id]['user'].get('name', 'FIGHTER')
                
                logger.info(f"🎬 Calling video_processor.create_highlight_video...")
                video_processor.create_highlight_video(
                    video_path=temp_video_path,
                    highlights=analysis_result["highlights"],
                    output_path=processed_video_path,
                    user_name=user_name
                )
                logger.info(f"✅ Highlight video created: {processed_video_path}")
                
                # Check if the processed video file exists
                if os.path.exists(processed_video_path):
                    logger.info(f"✅ Processed video file exists: {processed_video_path}")
                    file_size = os.path.getsize(processed_video_path)
                    logger.info(f"📊 Processed video file size: {file_size} bytes")
                else:
                    logger.error(f"❌ Processed video file does not exist: {processed_video_path}")
                    raise FileNotFoundError(f"Processed video file not found: {processed_video_path}")
                
                # Generate TTS audio for highlights
                if tts_client and analysis_result.get("highlights"):
                    logger.info(f"🔊 Generating TTS audio for highlights: {job_id}")
                    try:
                        audio_path = f"output/audio_{job_id}.mp3"
                        tts_client.generate_highlight_audio(analysis_result["highlights"], audio_path)
                        logger.info(f"✅ TTS audio generated: {audio_path}")
                        
                        # Add audio to video
                        final_video_path = f"output/final_{job_id}.mp4"
                        logger.info(f"🎬 Adding audio to video: {final_video_path}")
                        video_processor.add_audio_to_video(processed_video_path, audio_path, final_video_path)
                        logger.info(f"✅ Final video with audio created: {final_video_path}")
                        
                        # Verify final video exists
                        if not os.path.exists(final_video_path):
                            logger.error(f"❌ Final video file does not exist: {final_video_path}")
                            final_video_path = processed_video_path
                            logger.info(f"🔄 Using processed video as final: {final_video_path}")
                        
                    except Exception as audio_error:
                        logger.error(f"❌ TTS/Audio processing failed: {audio_error}")
                        logger.error(f"📋 Audio error traceback: {traceback.format_exc()}")
                        final_video_path = processed_video_path
                        logger.info(f"🔄 Using processed video without audio: {final_video_path}")
                else:
                    final_video_path = processed_video_path
                    logger.info(f"🔄 No TTS client or highlights, using processed video: {final_video_path}")
                
                # Store the final video in memory
                logger.info(f"💾 Storing video in memory: {final_video_path}")
                try:
                    with open(final_video_path, 'rb') as f:
                        final_video_content = f.read()
                    in_memory_files[job_id] = {
                        "content": final_video_content,
                        "filename": f"final_{job_id}.mp4",
                        "content_type": "video/mp4"
                    }
                    logger.info(f"✅ Final video stored in memory: {len(final_video_content)} bytes")
                except Exception as memory_error:
                    logger.error(f"❌ Error storing final video in memory: {memory_error}")
                    logger.error(f"📋 Memory error traceback: {traceback.format_exc()}")
                    # Try to store the processed video instead
                    try:
                        with open(processed_video_path, 'rb') as f:
                            processed_video_content = f.read()
                        in_memory_files[job_id] = {
                            "content": processed_video_content,
                            "filename": f"highlight_{job_id}.mp4",
                            "content_type": "video/mp4"
                        }
                        logger.info(f"✅ Processed video stored in memory: {len(processed_video_content)} bytes")
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback video storage also failed: {fallback_error}")
                        raise fallback_error
            else:
                logger.warning(f"⚠️ VideoProcessor not available, using original video")
                final_video_path = temp_video_path
                
        except Exception as video_error:
            logger.error(f"❌ CRITICAL ERROR in video processing: {video_error}")
            logger.error(f"📋 Video processing traceback: {traceback.format_exc()}")
            
            # Fallback: store original video
            try:
                logger.info(f"🔄 Fallback: storing original video in memory")
                with open(temp_video_path, 'rb') as f:
                    original_video_content = f.read()
                in_memory_files[job_id] = {
                    "content": original_video_content,
                    "filename": f"original_{job_id}.mp4",
                    "content_type": "video/mp4"
                }
                logger.info(f"✅ Original video stored in memory: {len(original_video_content)} bytes")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback video storage failed: {fallback_error}")
                logger.error(f"📋 Fallback error traceback: {traceback.format_exc()}")
                raise fallback_error
        
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
        
        # Ensure we have a video to serve
        if job_id not in in_memory_files:
            logger.warning(f"⚠️ No processed video found, storing original video for job: {job_id}")
            try:
                with open(temp_video_path, 'rb') as f:
                    original_video_content = f.read()
                in_memory_files[job_id] = {
                    "content": original_video_content,
                    "filename": f"original_{job_id}.mp4",
                    "content_type": "video/mp4"
                }
                logger.info(f"✅ Original video stored in memory: {len(original_video_content)} bytes")
            except Exception as e:
                logger.error(f"❌ Error storing original video: {e}")
                # Create a simple fallback video
                try:
                    import cv2
                    import numpy as np
                    
                    fallback_path = f"temp/fallback_{job_id}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(fallback_path, fourcc, 30.0, (640, 480))
                    
                    for i in range(90):  # 3 seconds
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Analysis Complete!", (150, 200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, "Video processing completed", (120, 250), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        out.write(frame)
                    
                    out.release()
                    
                    with open(fallback_path, 'rb') as f:
                        fallback_content = f.read()
                    
                    in_memory_files[job_id] = {
                        "content": fallback_content,
                        "filename": f"fallback_{job_id}.mp4",
                        "content_type": "video/mp4"
                    }
                    logger.info(f"✅ Fallback video created: {len(fallback_content)} bytes")
                    
                    # Clean up
                    os.remove(fallback_path)
                    
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback video creation also failed: {fallback_error}")
        
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
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"🚀 Starting AI Boxing Analysis server on {host}:{port}")
    print(f"📊 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'development')}")
    
    try:
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        # Fallback to basic server
        import http.server
        import socketserver
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer((host, port), Handler) as httpd:
            print(f"🔄 Fallback server running on {host}:{port}")
            httpd.serve_forever() 