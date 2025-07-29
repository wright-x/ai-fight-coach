"""
TTS Client - Railway Compatible Version
Generates speech using ElevenLabs API
"""

import os
import time
from typing import Optional

class TTSClient:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if self.api_key:
            try:
                import elevenlabs
                elevenlabs.set_api_key(self.api_key)
                self.elevenlabs = elevenlabs
                print("✅ TTSClient initialized (ElevenLabs mode)")
            except ImportError as e:
                print(f"⚠️ ElevenLabs import failed: {e}")
                self.api_key = None
        else:
            print("⚠️ TTSClient initialized (no API key)")
    
    def generate_speech(self, text: str, output_path: str) -> str:
        """Generate speech from text using ElevenLabs"""
        try:
            if not self.api_key or not hasattr(self, 'elevenlabs'):
                print("⚠️ No ElevenLabs API key or import failed, skipping TTS")
                return output_path
            
            print(f"🔊 Generating speech for: {text[:50]}...")
            
            # Generate audio
            audio = self.elevenlabs.generate(
                text=text,
                voice="21m00Tcm4TlvDq8ikWAM",  # Use a default voice ID
                model="eleven_monolingual_v1"
            )
            
            # Save audio file
            self.elevenlabs.save(audio, output_path)
            
            print(f"✅ Speech generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return output_path
    
    def generate_highlight_audio(self, highlights: list, output_path: str) -> str:
        """Generate audio for all highlights"""
        try:
            if not self.api_key or not hasattr(self, 'elevenlabs'):
                print("⚠️ No ElevenLabs API key or import failed, skipping highlight audio")
                return output_path
            
            # Combine all highlights into one text using ONLY action_required
            combined_text = "Here are your boxing highlights. "
            
            for i, highlight in enumerate(highlights, 1):
                action = highlight.get('action_required', '')
                
                combined_text += f"Highlight {i}: {action}. "
            
            return self.generate_speech(combined_text, output_path)
            
        except Exception as e:
            print(f"Error generating highlight audio: {e}")
            return output_path 