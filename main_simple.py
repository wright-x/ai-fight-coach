"""
AI Fight Coach - Simplified Main FastAPI Application
Version with better error handling for deployment
"""

# Suppress TensorFlow/MediaPipe C++ warnings
import os
import json
import logging
import tempfile
import shutil
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from pydantic import BaseModel
import uuid

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    # name = Column(String, nullable=True)  # Temporarily disabled - column doesn't exist
    signup_ts = Column(DateTime, default=datetime.utcnow, nullable=False)
    upload_count = Column(Integer, default=0, nullable=False)
    # jobs = relationship("Job", back_populates="user")  # Temporarily disabled - relationship error

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_ts = Column(DateTime, default=datetime.utcnow, nullable=False)
    video_url = Column(String, nullable=True)
    status = Column(String, default="processing", nullable=False)
    viewed = Column(Boolean, default=False, nullable=False)  # Add missing viewed column
    # analysis_type = Column(String, default="general", nullable=False)  # Temporarily disabled - column doesn't exist
    # view_count = Column(Integer, default=0, nullable=False)  # Temporarily disabled - column doesn't exist
    # user = relationship("User", back_populates="jobs")  # Temporarily disabled - relationship error
    # views = relationship("JobView", back_populates="job")  # Temporarily disabled - relationship error

class JobView(Base):
    __tablename__ = "job_views"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    viewed_ts = Column(DateTime, default=datetime.utcnow, nullable=False)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    # job = relationship("Job", back_populates="views")  # Temporarily disabled - relationship error

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database service
class DatabaseService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, email: str, name: str = None) -> Optional[User]:
        """Create a new user or return existing one"""
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(User.email == email).first()
            if existing_user:
                logger.info(f"User already exists: {existing_user.id}")
                return existing_user  # Return existing user instead of None
            
            # Create new user (without name column)
            user = User(email=email)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"Created new user: {user.id}")
            return user
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            self.db.rollback()
            return None
    
    def create_job(self, user_id: int, video_url: str = None, analysis_type: str = "general") -> Job:
        """Create a new job"""
        try:
            job = Job(
                id=str(uuid.uuid4()),
                user_id=user_id,
                video_url=video_url,
                status="processing"
                # analysis_type column doesn't exist
            )
            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)
            return job
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            self.db.rollback()
            raise
    
    def update_job_status(self, job_id: str, status: str, video_url: str = None):
        """Update job status"""
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            if video_url:
                job.video_url = video_url
            self.db.commit()
    
    def record_job_view(self, job_id: str, ip_address: str = None, user_agent: str = None):
        """Record a job view"""
        view = JobView(job_id=job_id, ip_address=ip_address, user_agent=user_agent)
        self.db.add(view)
        self.db.commit()
    
    def get_user_stats(self, user_id: int):
        """Get user statistics"""
        jobs = self.db.query(Job).filter(Job.user_id == user_id).all()
        total_views = 0
        for job in jobs:
            views = self.db.query(JobView).filter(JobView.job_id == job.id).count()
            total_views += views
        
        return {
            "total_jobs": len(jobs),
            "total_views": total_views,
            "completed_jobs": len([j for j in jobs if j.status == "completed"])
        }
    
    def get_admin_stats(self):
        """Get admin statistics"""
        total_users = self.db.query(User).count()
        total_jobs = self.db.query(Job).count()
        total_views = self.db.query(JobView).count()
        
        return {
            "total_users": total_users,
            "total_jobs": total_jobs,
            "total_views": total_views
        }

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
try:
    from utils.video_processor import VideoProcessor
    from utils.gemini_client import GeminiClient
    from utils.tts_client import TTSClient
    
    video_processor = VideoProcessor()
    gemini_client = GeminiClient()
    tts_client = TTSClient()
    
    logger.info("✅ Components initialized:")
    logger.info(f"   - VideoProcessor: {type(video_processor)}")
    logger.info(f"   - GeminiClient: {type(gemini_client)}")
    logger.info(f"   - TTSClient: {type(tts_client)}")
    
except Exception as e:
    logger.error(f"❌ Component initialization failed: {e}")
    raise

# In-memory storage
in_memory_files = {}
active_jobs = {}

app = FastAPI(title="AI Fight Coach", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

# Admin token verification
def verify_admin_token(request: Request):
    admin_token = os.getenv("ADMIN_TOKEN", "admin123")
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    return token == admin_token

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Root page - Registration
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/register.html")

# Main upload page
@app.get("/main", response_class=HTMLResponse)
async def main_page():
    return FileResponse("static/index.html")

# Registration endpoint
@app.post("/register")
async def register_user(request: Request, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        logger.info("Registration attempt started")
        
        body = await request.json()
        name = body.get("name", "Anonymous")  # Default name since column doesn't exist
        email = body.get("email")
        
        logger.info(f"Registration data: name={name}, email={email}")
        
        if not email:
            logger.error("Registration failed: Missing email")
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Test database connection
        try:
            db.execute(text("SELECT 1"))
            logger.info("Database connection successful")
        except Exception as db_error:
            logger.error(f"Database connection failed: {db_error}")
            return JSONResponse({
                "success": False,
                "message": "Database connection failed. Please try again."
            })
        
        db_service = DatabaseService(db)
        user = db_service.create_user(email=email, name=name)
        
        if not user:
            logger.error(f"User creation failed for email: {email}")
            return JSONResponse({
                "success": False,
                "message": "Database error. Please try again."
            })
        
        logger.info(f"User processed successfully: {user.id}")
        
        # Set cookies for 30 days (for both new and existing users)
        response = JSONResponse({
            "success": True,
            "message": f"Welcome {name}! Your account is ready."
        })
        
        response.set_cookie(
            key="user_email",
            value=email,
            max_age=2592000,  # 30 days
            path="/"
        )
        response.set_cookie(
            key="user_name", 
            value=name,
            max_age=2592000,
            path="/"
        )
        response.set_cookie(
            key="user_registered",
            value="true",
            max_age=2592000,
            path="/"
        )
        
        logger.info("Registration completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": "Registration failed. Please try again."
        })

# Admin dashboard
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    response = FileResponse("static/admin.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["ETag"] = f'"{int(datetime.utcnow().timestamp())}"'
    return response

@app.get("/api/admin/stats")
async def admin_stats(db: Session = Depends(get_db), _: bool = Depends(verify_admin_token)):
    """Get admin statistics"""
    try:
        logger.info("Admin stats request started")
        
        # Test database connection
        try:
            db.execute(text("SELECT 1"))
            logger.info("Database connection successful for admin stats")
        except Exception as db_error:
            logger.error(f"Database connection failed for admin stats: {db_error}")
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        db_service = DatabaseService(db)
        stats = db_service.get_admin_stats()
        logger.info(f"Admin stats retrieved: {stats}")
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/users")
async def admin_users(db: Session = Depends(get_db), _: bool = Depends(verify_admin_token)):
    """Get all users with stats"""
    try:
        logger.info("Admin users request started")
        
        # Test database connection
        try:
            db.execute(text("SELECT 1"))
            logger.info("Database connection successful for admin users")
        except Exception as db_error:
            logger.error(f"Database connection failed for admin users: {db_error}")
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        users = db.query(User).all()
        logger.info(f"Found {len(users)} users")
        
        result = []
        for user in users:
            db_service = DatabaseService(db)
            stats = db_service.get_user_stats(user.id)
            result.append({
                "id": user.id,
                "email": user.email,
                "name": "Anonymous",  # Default since column doesn't exist
                "signup_ts": user.signup_ts.isoformat(),
                "upload_count": user.upload_count,
                "total_jobs": stats["total_jobs"],
                "total_views": stats["total_views"],
                "completed_jobs": stats["completed_jobs"]
            })
        
        logger.info(f"Admin users data prepared: {len(result)} users")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin users error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/jobs")
async def admin_jobs(db: Session = Depends(get_db), _: bool = Depends(verify_admin_token)):
    """Get all jobs with view counts"""
    try:
        logger.info("Admin jobs request started")
        
        # Test database connection
        try:
            db.execute(text("SELECT 1"))
            logger.info("Database connection successful for admin jobs")
        except Exception as db_error:
            logger.error(f"Database connection failed for admin jobs: {db_error}")
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        jobs = db.query(Job).all()
        logger.info(f"Found {len(jobs)} jobs")
        
        result = []
        for job in jobs:
            view_count = db.query(JobView).filter(JobView.job_id == job.id).count()
            result.append({
                "id": job.id,
                "user_id": job.user_id,
                "created_ts": job.created_ts.isoformat(),
                "status": job.status,
                "analysis_type": "general",  # Default since column doesn't exist
                "video_url": job.video_url,
                "view_count": view_count
            })
        
        logger.info(f"Admin jobs data prepared: {len(result)} jobs")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin jobs error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Video upload endpoint
@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    email: str = Form(...),
    analysis_type: str = Form("general"),
    db: Session = Depends(get_db)
):
    """Upload and process video"""
    try:
        logger.info(f"Upload request started: email={email}, analysis_type={analysis_type}")
        logger.info(f"File details: filename={file.filename}, content_type={file.content_type}")
        
        # Validate file
        if not file.filename:
            logger.error("Upload failed: No filename provided")
            return JSONResponse({
                "success": False,
                "message": "No file provided"
            }, status_code=422)
        
        # Validate file type
        allowed_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        file_ext = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: {file_ext}")
        
        if file_ext not in allowed_extensions:
            logger.error(f"Upload failed: Invalid file type {file_ext}")
            return JSONResponse({
                "success": False,
                "message": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            }, status_code=422)
        
        # Create or get user
        logger.info("Creating/getting user...")
        db_service = DatabaseService(db)
        user = db_service.create_user(email=email)
        
        if not user:
            logger.error(f"Upload failed: Could not create/get user for {email}")
            return JSONResponse({
                "success": False,
                "message": "User creation failed"
            }, status_code=500)
        
        logger.info(f"User processed: {user.id}")
        
        # Create job
        logger.info("Creating job...")
        job = db_service.create_job(user.id, analysis_type=analysis_type)
        
        if not job:
            logger.error(f"Upload failed: Could not create job for user {user.id}")
            return JSONResponse({
                "success": False,
                "message": "Job creation failed"
            }, status_code=500)
        
        logger.info(f"Job created: {job.id}")
        
        # Save uploaded file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = f"{temp_dir}/{job.id}_{file.filename}"
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File saved to {file_path}")
        except Exception as e:
            logger.error(f"File save failed: {e}")
            return JSONResponse({
                "success": False,
                "message": "File save failed"
            }, status_code=500)
        
        # Store job data
        in_memory_files[job.id] = {
            "path": file_path,
            "analysis_type": analysis_type
        }
        
        # Start background processing (synchronous for now)
        logger.info(f"Starting video processing for job {job.id}")
        try:
            await process_video_analysis(job.id, db)
            logger.info(f"Video processing completed for job {job.id}")
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            # Don't fail the upload, just log the error
        
        logger.info(f"Job {job.id} created successfully for user {email}")
        
        return JSONResponse({
            "success": True,
            "job_id": job.id,
            "message": "Video uploaded successfully"
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return JSONResponse({
            "success": False,
            "message": f"Upload failed: Unknown error"
        }, status_code=500)

# Status endpoint
@app.get("/status/{job_id}")
async def get_status(job_id: str, db: Session = Depends(get_db)):
    """Get job status"""
    try:
        db_service = DatabaseService(db)
        job = db.query(Job).filter(Job.id == job_id).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job.id,
            "status": job.status,
            "video_url": job.video_url,
            "created_ts": job.created_ts.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Video serving
@app.get("/video/{job_id}")
async def get_video(job_id: str):
    """Serve processed video"""
    video_path = f"output/highlight_{job_id}.mp4"
    if os.path.exists(video_path):
        return FileResponse(video_path)
    else:
        raise HTTPException(status_code=404, detail="Video not found")

# Analysis data
@app.get("/analysis/{job_id}")
async def get_analysis(job_id: str):
    """Get analysis results"""
    analysis_path = f"output/analysis_{job_id}.json"
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=404, detail="Analysis not found")

# Results page
@app.get("/results/{job_id}", response_class=HTMLResponse)
async def get_results(job_id: str, db: Session = Depends(get_db)):
    """Show results page"""
    try:
        # Record view
        db_service = DatabaseService(db)
        db_service.record_job_view(job_id)
        
        return FileResponse("static/index.html")
        
    except Exception as e:
        logger.error(f"Results error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Video processing
async def process_video_analysis(job_id: str, db: Session):
    """Process video analysis"""
    try:
        logger.info(f"Starting analysis for job {job_id}")
        
        # Get job data
        job_data = in_memory_files.get(job_id)
        if not job_data:
            logger.error(f"No job data found for {job_id}")
            return
        
        # Update job status
        db_service = DatabaseService(db)
        db_service.update_job_status(job_id, "processing")
        
        # Process video
        video_path = job_data["path"]
        analysis_type = job_data["analysis_type"]
        
        # Create output paths
        output_video_path = f"output/highlight_{job_id}.mp4"
        output_analysis_path = f"output/analysis_{job_id}.json"
        
        os.makedirs("output", exist_ok=True)
        
        # Analyze video with Gemini
        analysis_result = gemini_client.analyze_video(video_path, analysis_type)
                logger.info(json.dumps(analysis_result, indent=2))
        
        # Save analysis results
        with open(output_analysis_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        # Generate highlight video
        highlights = analysis_result.get('highlights', [])
        if highlights:
            try:
                logger.info(f"Creating highlight video for job {job_id}")
                video_processor.create_highlight_video(
                    video_path=video_path,
                    highlights=highlights,
                    output_path=output_video_path,
                    user_name="FIGHTER"
                )
                logger.info(f"Highlight video created successfully: {output_video_path}")
            except Exception as video_error:
                logger.error(f"Video creation failed for job {job_id}: {video_error}")
                else:
            logger.warning(f"No highlights found for job {job_id}")
                
        # Update job status
        video_url = f"/video/{job_id}" if os.path.exists(output_video_path) else None
        db_service.update_job_status(job_id, "completed", video_url)
                        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        
        logger.info(f"Analysis completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {e}")
        try:
            db_service = DatabaseService(db)
            db_service.update_job_status(job_id, "failed")
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 