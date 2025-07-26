import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Import our modules
from llm_client import GeminiClient
from utils.tts_utils import generate_tts
from utils.overlay_utils import add_feedback_overlay_with_head_tracking
from utils.video_utils import combine_videos_async, add_audio_to_video
from analyzer import create_highlights_section

# Global sanitizer import
import log_sanitizer

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Directory setup
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
DEBUG_DIR = BASE_DIR / "debug_logs"
STATIC_DIR = BASE_DIR / "static"
PROMPTS_DIR = BASE_DIR / "app" / "prompts"
TEMP_DIR = BASE_DIR / "temp"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

# Create all directories
for directory in [UPLOAD_DIR, OUTPUT_DIR, DEBUG_DIR, STATIC_DIR, PROMPTS_DIR, TEMP_DIR, TEMPLATES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {directory}")

# Initialize Gemini client
gemini_client = GeminiClient()

app = FastAPI(title="AI Fight Coach", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...), prompt: str = None):
    """Analyze uploaded video with Gemini and generate feedback"""
    try:
        # Save uploaded video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = UPLOAD_DIR / f"input_{timestamp}.mp4"
        
        with open(input_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        logger.info(f"Saved uploaded video to {input_path}")
        
        # Load default prompt if none provided
        if not prompt:
            try:
                default_prompt_path = PROMPTS_DIR / "default.txt"
                with open(default_prompt_path, "r") as f:
                    prompt = f.read().strip()
                logger.info(f"Loaded default prompt from {default_prompt_path}")
            except FileNotFoundError:
                prompt = "Analyze this boxing video and provide feedback."
                logger.warning(f"Default prompt file not found, using fallback")
        
        final_prompt = prompt
        logger.info(f"Using prompt: {final_prompt}")
        
        # Send to Gemini
        logger.info(f"Sending to Gemini: video={input_path}, prompt={final_prompt}")
        response = await gemini_client.analyze_video(input_path, final_prompt)
        
        # Log Gemini response without binary data
        debug_log_path = DEBUG_DIR / f"gemini_response_{timestamp}.json"
        with open(debug_log_path, "w") as f:
            json.dump({
                "prompt": final_prompt,
                "video_path": str(input_path),
                "response": response
            }, f, indent=2)
        logger.info(f"Saved Gemini response to {debug_log_path}")
        
        # Generate TTS with synchronized timing
        logger.info("Generating TTS with synchronized timing...")
        tts_path = await generate_tts(response, OUTPUT_DIR / f"tts_{timestamp}.mp3")
        
        # Add feedback overlay with synchronized captions
        logger.info("Adding feedback overlay with synchronized captions...")
        overlay_path = await add_feedback_overlay_with_head_tracking(
            input_path, 
            response, 
            tts_path,
            OUTPUT_DIR / f"overlay_{timestamp}.mp4"
        )
        
        # Create highlights section
        logger.info("Creating highlights section...")
        highlights_path = await create_highlights_section(
            input_path,
            response,
            OUTPUT_DIR / f"highlights_{timestamp}.mp4"
        )
        
        # Add audio to both videos
        logger.info("Adding audio to videos...")
        overlay_with_audio = await add_audio_to_video(
            overlay_path, 
            tts_path, 
            OUTPUT_DIR / f"overlay_audio_{timestamp}.mp4"
        )
        
        highlights_with_audio = await add_audio_to_video(
            highlights_path, 
            tts_path, 
            OUTPUT_DIR / f"highlights_audio_{timestamp}.mp4"
        )
        
        # Combine videos
        logger.info("Combining videos...")
        final_path = await combine_videos_async(
            overlay_with_audio,
            highlights_with_audio,
            OUTPUT_DIR / f"final_{timestamp}.mp4"
        )
        
        # Cleanup temporary files
        for temp_file in [input_path, tts_path, overlay_path, highlights_path, 
                         overlay_with_audio, highlights_with_audio]:
            if temp_file.exists():
                temp_file.unlink()
                logger.info(f"Cleaned up: {temp_file}")
        
        # Schedule final cleanup
        asyncio.create_task(schedule_cleanup(final_path, delay=300))  # 5 minutes
        
        return {"status": "success", "video_path": str(final_path)}
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        # Cleanup on error
        for temp_file in [input_path, tts_path, overlay_path, highlights_path, 
                         overlay_with_audio, highlights_with_audio]:
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

async def schedule_cleanup(file_path: Path, delay: int = 300):
    """Schedule file cleanup after delay"""
    await asyncio.sleep(delay)
    if file_path.exists():
        file_path.unlink()
        logger.info(f"Scheduled cleanup: {file_path}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 