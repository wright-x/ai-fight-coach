"""
Text-to-Speech client using ElevenLabs for AI Fight Coach application.
Generates audio from Gemini's textual analysis output.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List
from elevenlabs import generate, save, set_api_key, voices
from elevenlabs.api import History
from .logger import logger


class TTSClient:
    """Client for generating TTS audio using ElevenLabs."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable.")
        
        set_api_key(self.api_key)
        self.available_voices = self._get_available_voices()
        logger.info(f"TTSClient initialized with {len(self.available_voices)} available voices")
    
    def _get_available_voices(self) -> List[dict]:
        """Get list of available ElevenLabs voices."""
        try:
            voices_list = voices()
            return [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category
                }
                for voice in voices_list
            ]
        except Exception as e:
            logger.warning(f"Could not fetch available voices: {e}")
            # Return some default voices that should work
            return [
                {
                    "voice_id": "21m00Tcm4TlvDq8ikWAM",
                    "name": "Rachel",
                    "category": "default"
                },
                {
                    "voice_id": "AZnzlk1XvdvUeBnXmlld",
                    "name": "Domi",
                    "category": "default"
                }
            ]
    
    def get_default_voice(self) -> str:
        """Get a default voice ID for TTS generation."""
        # Use the specified voice ID
        return "YXpFCvM1S3JbWEJhoskW"
    
    def generate_highlights_audio(self, highlights: List[dict], 
                                voice_id: Optional[str] = None) -> str:
        """
        Generate TTS audio for highlights.
        
        Args:
            highlights: List of highlight items from Gemini analysis
            voice_id: Optional voice ID to use
            
        Returns:
            Path to generated audio file
        """
        try:
            logger.info(f"Generating highlights audio for {len(highlights)} highlights")
            
            # Prepare text content
            text_content = self._prepare_highlights_text(highlights)
            
            # Generate audio
            audio_path = self._generate_audio(text_content, voice_id, "highlights")
            
            logger.info(f"Highlights audio generated: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error generating highlights audio: {e}")
            raise
    
    def generate_full_analysis_audio(self, analysis_result: dict, 
                                   voice_id: Optional[str] = None) -> str:
        """
        Generate complete TTS audio for the entire analysis.
        
        Args:
            analysis_result: Complete analysis result from Gemini
            voice_id: Optional voice ID to use
            
        Returns:
            Path to generated audio file
        """
        try:
            logger.info("Generating full analysis audio")
            
            # Prepare complete text content
            text_content = self._prepare_full_analysis_text(analysis_result)
            
            # Generate audio
            audio_path = self._generate_audio(text_content, voice_id, "full_analysis")
            
            logger.info(f"Full analysis audio generated: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error generating full analysis audio: {e}")
            raise
    
    def _prepare_highlights_text(self, highlights: List[dict]) -> str:
        """Prepare text content for highlights with 800ms pauses."""
        text_parts = ["Key Highlights:"]
        
        for i, highlight in enumerate(highlights, 1):
            detailed_feedback = highlight.get('detailed_feedback', '')
            action_required = highlight.get('action_required', '')
            text_parts.append(f"Highlight {i}: {detailed_feedback}")
            if action_required:
                text_parts.append(f"Action required: {action_required}")
            text_parts.append("[800ms]")  # 800ms pause after each highlight
        
        return " ".join(text_parts)
    
    def _prepare_full_analysis_text(self, analysis_result: dict) -> str:
        """Prepare complete text content for full analysis with 800ms pauses between sentences."""
        text_parts = []
        
        # Highlights - add clear separation with pauses
        highlights = analysis_result.get('highlights', [])
        if highlights:
            text_parts.append("[800ms] Key highlights [800ms]")  # Clear separator with pauses
            for i, highlight in enumerate(highlights, 1):
                detailed_feedback = highlight.get('detailed_feedback', '')
                action_required = highlight.get('action_required', '')
                text_parts.append(f"Highlight {i}: {detailed_feedback}")
                if action_required:
                    text_parts.append(f"Action: {action_required}")
                text_parts.append("[800ms]")  # 800ms pause between highlights
        
        return " ".join(text_parts)
    
    def _generate_audio(self, text: str, voice_id: Optional[str], 
                       audio_type: str) -> str:
        """Generate audio from text using ElevenLabs."""
        try:
            # Use default voice if none specified
            if not voice_id:
                voice_id = self.get_default_voice()
            
            logger.debug(f"Generating audio with voice {voice_id} for {audio_type}")
            
            # Generate audio
            audio = generate(
                text=text,
                voice=voice_id,
                model="eleven_monolingual_v1"
            )
            
            # Save audio to file
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            audio_filename = f"{audio_type}_{voice_id}.mp3"
            audio_path = output_dir / audio_filename
            
            # Handle both bytes and iterator
            if hasattr(audio, '__iter__') and not isinstance(audio, bytes):
                audio_bytes = b''.join(audio)
            else:
                audio_bytes = audio
                
            save(audio_bytes, str(audio_path))
            
            logger.debug(f"Audio saved to: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise
    
    def get_voice_info(self, voice_id: str) -> Optional[dict]:
        """Get information about a specific voice."""
        for voice in self.available_voices:
            if voice["voice_id"] == voice_id:
                return voice
        return None
    
    def list_voices(self) -> List[dict]:
        """Get list of all available voices."""
        return self.available_voices.copy() 