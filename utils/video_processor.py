"""
Video Processor - Professional Highlight Reel System (Simplified)
Creates cinematic highlight videos with captions and transitions
"""

import os
import shutil
from typing import List, Dict, Any, Optional
import tempfile
import time
import traceback
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, TextClip, concatenate_videoclips

class VideoProcessor:
    """Professional video processor for creating highlight reels"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        print("✅ VideoProcessor initialized (Professional mode)")
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str) -> str:
        """Create professional highlight reel with captions and transitions"""
        try:
            print(f"🎬 Creating professional highlight reel: {video_path}")
            print(f"📊 Number of highlights: {len(highlights)}")
            
            # Load original video
            video = VideoFileClip(video_path)
            duration = video.duration
            fps = video.fps
            print(f"📊 Video: {duration}s at {fps}fps")
            
            # Create clips list
            all_clips = []
            
            # Add intro screen
            intro_clip = self._create_intro_screen(duration=3)
            all_clips.append(intro_clip)
            
            # Process each highlight
            for i, highlight in enumerate(highlights):
                print(f"🎯 Processing highlight {i+1}/{len(highlights)}")
                
                # Create highlight transition screen
                if i > 0:  # Skip for first highlight
                    transition_clip = self._create_transition_screen(f"HIGHLIGHT {i+1}", duration=2)
                    all_clips.append(transition_clip)
                
                # Create highlight clip with captions
                highlight_clip = self._create_highlight_clip(video, highlight, i+1)
                all_clips.append(highlight_clip)
            
            # Concatenate all clips
            print(f"🎬 Concatenating {len(all_clips)} clips...")
            final_video = concatenate_videoclips(all_clips, method="compose")
            
            # Write final video
            print(f"💾 Writing final highlight reel...")
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
            for clip in all_clips:
                clip.close()
            
            print(f"✅ Professional highlight reel created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating highlight reel: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            # Fallback to simple copy
            shutil.copy2(video_path, output_path)
            return output_path
    
    def _create_intro_screen(self, duration: float = 3.0) -> VideoFileClip:
        """Create beautiful intro screen"""
        try:
            # Create black background
            bg = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
            
            # Create title text
            title = TextClip(
                "AI BOXING ANALYSIS",
                fontsize=120,
                color='white',
                font='Arial-Bold',
                stroke_color='#FF6B35',
                stroke_width=3
            ).set_position('center').set_duration(duration)
            
            # Create subtitle
            subtitle = TextClip(
                "Professional Fight Analysis",
                fontsize=60,
                color='#FF6B35',
                font='Arial'
            ).set_position(('center', 600)).set_duration(duration)
            
            # Create composite
            intro = CompositeVideoClip([bg, title, subtitle])
            return intro
            
        except Exception as e:
            print(f"⚠️ Intro screen failed: {e}")
            # Fallback to simple black screen
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _create_transition_screen(self, text: str, duration: float = 2.0) -> VideoFileClip:
        """Create transition screen between highlights"""
        try:
            # Create black background
            bg = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
            
            # Create highlight text
            highlight_text = TextClip(
                text,
                fontsize=100,
                color='#FF6B35',
                font='Arial-Bold',
                stroke_color='white',
                stroke_width=2
            ).set_position('center').set_duration(duration)
            
            # Create composite
            transition = CompositeVideoClip([bg, highlight_text])
            return transition
            
        except Exception as e:
            print(f"⚠️ Transition screen failed: {e}")
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _create_highlight_clip(self, video: VideoFileClip, highlight: Dict, highlight_num: int) -> VideoFileClip:
        """Create individual highlight clip with captions"""
        try:
            timestamp = highlight.get('timestamp', '00:00')
            feedback = highlight.get('detailed_feedback', 'No feedback')
            action = highlight.get('action_required', 'No action')
            problem_location = highlight.get('problem_location', 'general')
            
            # Parse timestamp
            highlight_time = self._parse_timestamp(timestamp)
            
            # Calculate clip timing (3 seconds before/after)
            start_time = max(0, highlight_time - 3)
            end_time = min(video.duration, highlight_time + 3)
            clip_duration = end_time - start_time
            
            print(f"🎯 Highlight {highlight_num}: {timestamp} ({start_time}s - {end_time}s)")
            
            # Extract video clip
            video_clip = video.subclip(start_time, end_time)
            
            # Slow down the clip (0.5x speed)
            slow_clip = video_clip.speedx(0.5)
            
            # Create analysis frame (3 seconds)
            analysis_clip = self._create_analysis_frame(feedback, problem_location, 3)
            
            # Combine analysis frame + slow motion clip
            final_clip = concatenate_videoclips([analysis_clip, slow_clip], method="compose")
            
            # Add captions
            caption_clip = self._create_caption_overlay(feedback, action, final_clip.duration)
            final_clip = CompositeVideoClip([final_clip, caption_clip])
            
            return final_clip
            
        except Exception as e:
            print(f"⚠️ Highlight clip failed: {e}")
            # Fallback to simple subclip
            highlight_time = self._parse_timestamp(highlight.get('timestamp', '00:00'))
            start_time = max(0, highlight_time - 3)
            end_time = min(video.duration, highlight_time + 3)
            return video.subclip(start_time, end_time)
    
    def _create_analysis_frame(self, feedback: str, problem_location: str, duration: float) -> VideoFileClip:
        """Create analysis frame with text overlay"""
        try:
            # Create black background
            bg = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
            
            # Create problem location text
            location_text = TextClip(
                f"PROBLEM: {problem_location.upper()}",
                fontsize=80,
                color='#FF6B35',
                font='Arial-Bold'
            ).set_position(('center', 300)).set_duration(duration)
            
            # Create feedback text (wrapped)
            feedback_text = TextClip(
                feedback[:100] + "..." if len(feedback) > 100 else feedback,
                fontsize=50,
                color='white',
                font='Arial',
                method='caption',
                size=(1600, 200)
            ).set_position(('center', 500)).set_duration(duration)
            
            # Create composite
            analysis = CompositeVideoClip([bg, location_text, feedback_text])
            return analysis
            
        except Exception as e:
            print(f"⚠️ Analysis frame failed: {e}")
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _create_caption_overlay(self, feedback: str, action: str, duration: float) -> VideoFileClip:
        """Create caption overlay for highlight clip"""
        try:
            # Create background banner
            banner = ColorClip(size=(1920, 150), color=(0, 0, 0, 180), duration=duration)
            banner = banner.set_position(('center', 'bottom'))
            
            # Create feedback text
            feedback_text = TextClip(
                feedback[:100] + "..." if len(feedback) > 100 else feedback,
                fontsize=40,
                color='white',
                font='Arial',
                method='caption',
                size=(1800, 80)
            ).set_position(('center', 1080-140)).set_duration(duration)
            
            # Create action text
            action_text = TextClip(
                f"Action: {action}",
                fontsize=30,
                color='#FF6B35',
                font='Arial-Bold'
            ).set_position(('center', 1080-50)).set_duration(duration)
            
            # Combine
            captions = CompositeVideoClip([banner, feedback_text, action_text])
            return captions
            
        except Exception as e:
            print(f"⚠️ Caption overlay failed: {e}")
            return ColorClip(size=(1, 1), color=(0, 0, 0), duration=duration)
    
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
                video = VideoFileClip(video_path)
                audio = VideoFileClip(audio_path)
                
                # Combine video and audio
                final_video = video.set_audio(audio)
                final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
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