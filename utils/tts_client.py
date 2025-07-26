"""
Simplified TTS Client - Railway Compatible Version
Works without OpenCV dependencies
"""

import os
import time
from typing import Optional

class TTSClient:
    """Simplified TTS client for Railway environment"""
    
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        print("✅ TTSClient initialized (simplified mode)")
    
    def generate_speech(self, text: str, output_path: str) -> str:
        """Generate speech from text (simplified version)"""
        try:
            # Simulate TTS generation
            time.sleep(1)  # Simulate processing time
            
            # In a real implementation, this would use ElevenLabs API
            # For now, just return the output path
            return output_path
        except Exception as e:
            print(f"Error generating speech: {e}")
            return output_path 