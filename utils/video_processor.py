"""
Video Processor - Railway Compatible Version
Handles video editing and overlays using MoviePy (no OpenCV)
"""

import os
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ColorClip
from typing import List, Dict, Any, Optional
import tempfile
import time

class VideoProcessor:
    """Video processor for Railway environment using MoviePy"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        print("✅ VideoProcessor initialized (MoviePy mode)")
    
    def process_video(self, video_path: str, fighter_name: str = "FIGHTER") -> Dict[str, Any]:
        """Process video and return analysis results"""
        try:
            # This will be handled by Gemini client
            return {"status": "ready_for_analysis"}
        except Exception as e:
            print(f"Error processing video: {e}")
            return {"status": "error", "message": str(e)}
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str) -> str:
        """Create highlight video with overlays using MoviePy"""
        try:
            print(f"🎬 Creating highlight video: {video_path}")
            
            # Load video
            video = VideoFileClip(video_path)
            duration = video.duration
            
            # Create clips for each highlight
            highlight_clips = []
            
            for highlight in highlights:
                timestamp = self._parse_timestamp(highlight.get('timestamp', '0'))
                if timestamp < duration:
                    # Create text overlay for this highlight
                    feedback = highlight.get('detailed_feedback', '')
                    action = highlight.get('action_required', '')
                    
                    # Create text clips
                    feedback_clip = TextClip(
                        feedback, 
                        fontsize=24, 
                        color='white', 
                        font='Arial-Bold',
                        size=(video.w - 40, None),
                        method='caption'
                    ).set_position(('center', 50)).set_duration(5)
                    
                    action_clip = TextClip(
                        f"Action: {action}", 
                        fontsize=20, 
                        color='yellow', 
                        font='Arial-Bold'
                    ).set_position(('center', 150)).set_duration(5)
                    
                    # Create fighter name overlay
                    name_clip = TextClip(
                        "FIGHTER", 
                        fontsize=30, 
                        color='lime', 
                        font='Arial-Bold'
                    ).set_position(('center', video.h - 100)).set_duration(5)
                    
                    # Create highlight section
                    start_time = max(0, timestamp - 2)
                    end_time = min(duration, timestamp + 3)
                    
                    video_section = video.subclip(start_time, end_time)
                    
                    # Composite the clips
                    composite = CompositeVideoClip([
                        video_section,
                        feedback_clip.set_start(2),
                        action_clip.set_start(2),
                        name_clip.set_start(2)
                    ])
                    
                    highlight_clips.append(composite)
            
            # If no highlights, create a simple overlay version
            if not highlight_clips:
                # Add fighter name to entire video
                name_clip = TextClip(
                    "FIGHTER", 
                    fontsize=40, 
                    color='lime', 
                    font='Arial-Bold'
                ).set_position(('center', 50)).set_duration(duration)
                
                composite = CompositeVideoClip([video, name_clip])
                highlight_clips = [composite]
            
            # Concatenate all highlight clips
            if len(highlight_clips) > 1:
                final_video = CompositeVideoClip(highlight_clips)
            else:
                final_video = highlight_clips[0]
            
            # Write the final video
            final_video.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Clean up
            video.close()
            final_video.close()
            for clip in highlight_clips:
                clip.close()
            
            print(f"✅ Highlight video created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating highlight video: {e}")
            return video_path
    
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
        """Add audio to video using MoviePy"""
        try:
            print(f"🔊 Adding audio to video: {video_path}")
            
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # Combine video and audio
            final_video = video.set_audio(audio)
            final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            video.close()
            audio.close()
            final_video.close()
            
            print(f"✅ Audio added to video: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error adding audio to video: {e}")
            return video_path 