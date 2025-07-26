import os
import time
import logging
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from google.generativeai.types import Blob
from tenacity import retry, stop_after_attempt, wait_exponential
import cv2

# Configure logging
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        """Initialize Gemini client with API key"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        
        # Use the latest model
        self.model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp')
        logger.info(f"[LLM] Using Gemini model: {self.model.model_name}")
        
        logger.info("Initialized Gemini client")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_video(self, video_path: Path, prompt: str) -> str:
        """Analyze video with Gemini and return response"""
        try:
            logger.info("Starting video analysis with Gemini")
            logger.info(f"Video path: {video_path}")
            logger.info(f"Prompt: {prompt}")
            
            # Load video file as Blob
            with open(video_path, "rb") as f:
                video_data = f.read()
                blob = Blob(mime_type="video/mp4", data=video_data)
            logger.info(f"Loaded video file: {len(video_data)} bytes")
            
            # Generate response
            response = await self.model.generate_content_async([prompt, blob])
            
            # Extract response text
            response_text = response.text
            logger.info(f"Gemini response length: {len(response_text)} characters")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {type(e).__name__}: {str(e)[:500]}", exc_info=True)
            raise 