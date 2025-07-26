"""
Video Processor - Fast and Reliable Highlight System
Creates simple highlight videos without complex processing
"""

import os
import shutil
from typing import List, Dict, Any, Optional
import tempfile
import time
import traceback

class VideoProcessor:
    """Fast video processor for creating highlight reels"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        print("✅ VideoProcessor initialized (Fast mode)")
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str) -> str:
        """Create simple highlight video with basic overlays"""
        try:
            print(f"🎬 Creating fast highlight video: {video_path}")
            print(f"📊 Number of highlights: {len(highlights)}")
            
            # Simple approach: just copy the video and add a text overlay
            shutil.copy2(video_path, output_path)
            print(f"✅ Video copied to: {output_path}")
            
            # Try to add a simple overlay if MoviePy is available
            try:
                from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
                
                # Load video
                video = VideoFileClip(output_path)
                duration = video.duration
                print(f"📊 Video duration: {duration} seconds")
                
                # Create a simple text overlay
                overlay_text = f"AI ANALYSIS - {len(highlights)} HIGHLIGHTS"
                text_clip = TextClip(
                    overlay_text,
                    fontsize=60,
                    color='white',
                    font='Arial-Bold',
                    stroke_color='red',
                    stroke_width=3
                ).set_position(('center', 50)).set_duration(duration)
                
                # Create composite
                composite = CompositeVideoClip([video, text_clip])
                
                # Write with overlay
                composite.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                # Clean up
                video.close()
                composite.close()
                text_clip.close()
                
                print(f"✅ Highlight video created with overlay: {output_path}")
                
            except Exception as e:
                print(f"⚠️ MoviePy overlay failed, using copied video: {e}")
                # The video was already copied above, so we're good
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating highlight video: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            # Fallback to simple copy
            shutil.copy2(video_path, output_path)
            return output_path
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse timestamp string to seconds"""
        try:
            if ':' in timestamp:
                parts = timestamp.split(':')
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            else:
                return float(timestamp)
        except:
            return 0.0
    
    def add_audio_to_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Add TTS audio to video"""
        try:
            if os.path.exists(audio_path):
                from moviepy.editor import VideoFileClip, AudioFileClip
                
                video = VideoFileClip(video_path)
                audio = AudioFileClip(audio_path)
                
                # Combine video and audio
                final_video = video.set_audio(audio)
                final_video.write_videofile(
                    output_path, 
                    codec='libx264', 
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                
                video.close()
                audio.close()
                final_video.close()
                
                print(f"✅ Audio added to video: {output_path}")
                return output_path
            else:
                print(f"⚠️ Audio file not found: {audio_path}")
                shutil.copy2(video_path, output_path)
                return output_path
                
        except Exception as e:
            print(f"Error adding audio to video: {e}")
            shutil.copy2(video_path, output_path)
            return output_path 