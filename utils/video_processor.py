"""
Video Processor - Professional Highlight Reel System
Creates cinematic highlight videos with MediaPipe analysis, captions, and transitions
"""

import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import time
import traceback
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, TextClip, concatenate_videoclips
from moviepy.video.fx import resize

class VideoProcessor:
    """Professional video processor for creating highlight reels"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        print("✅ VideoProcessor initialized (Professional mode)")
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str) -> str:
        """Create professional highlight reel with MediaPipe analysis"""
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
                
                # Create highlight clip with MediaPipe analysis
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
        """Create individual highlight clip with MediaPipe analysis"""
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
            
            # Create MediaPipe analysis frame
            analysis_frame = self._create_mediapipe_analysis_frame(
                video_path=video.filename,
                timestamp=highlight_time,
                problem_location=problem_location,
                feedback=feedback
            )
            
            # Create analysis clip (3 seconds)
            analysis_clip = VideoFileClip(analysis_frame).set_duration(3)
            
            # Create caption overlay
            caption_clip = self._create_caption_overlay(feedback, action, clip_duration)
            
            # Combine analysis frame + slow motion clip
            final_clip = concatenate_videoclips([analysis_clip, slow_clip], method="compose")
            
            # Add captions
            final_clip = CompositeVideoClip([final_clip, caption_clip])
            
            return final_clip
            
        except Exception as e:
            print(f"⚠️ Highlight clip failed: {e}")
            # Fallback to simple subclip
            highlight_time = self._parse_timestamp(highlight.get('timestamp', '00:00'))
            start_time = max(0, highlight_time - 3)
            end_time = min(video.duration, highlight_time + 3)
            return video.subclip(start_time, end_time)
    
    def _create_mediapipe_analysis_frame(self, video_path: str, timestamp: float, problem_location: str, feedback: str) -> str:
        """Create MediaPipe analysis frame with body tracking and error pointer"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"⚠️ Could not read frame at {timestamp}s")
                return self._create_fallback_frame(feedback)
            
            # Initialize MediaPipe Pose
            with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            ) as pose:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                # Draw pose landmarks
                annotated_frame = frame.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Add error pointer based on problem location
                annotated_frame = self._add_error_pointer(annotated_frame, results.pose_landmarks, problem_location)
                
                # Add text overlay
                annotated_frame = self._add_text_overlay(annotated_frame, feedback, problem_location)
                
                # Save frame
                output_frame = f"temp/analysis_frame_{int(timestamp)}.jpg"
                cv2.imwrite(output_frame, annotated_frame)
                
                return output_frame
                
        except Exception as e:
            print(f"⚠️ MediaPipe analysis failed: {e}")
            return self._create_fallback_frame(feedback)
    
    def _add_error_pointer(self, frame: np.ndarray, landmarks, problem_location: str) -> np.ndarray:
        """Add error pointer arrow to specific body part"""
        try:
            if not landmarks:
                return frame
            
            # Define landmark indices for different body parts
            body_parts = {
                'head': 0,  # nose
                'left_hand': 19,  # left wrist
                'right_hand': 20,  # right wrist
                'left_shoulder': 11,
                'right_shoulder': 12,
                'left_elbow': 13,
                'right_elbow': 14,
                'left_hip': 23,
                'right_hip': 24,
                'left_knee': 25,
                'right_knee': 26,
                'left_foot': 31,
                'right_foot': 32
            }
            
            # Get landmark position
            if problem_location in body_parts:
                landmark_idx = body_parts[problem_location]
                landmark = landmarks.landmark[landmark_idx]
                
                # Convert to pixel coordinates
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Draw red arrow pointing to the problem area
                cv2.arrowedLine(frame, (x-50, y-50), (x, y), (0, 0, 255), 5, tipLength=0.3)
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                
            return frame
            
        except Exception as e:
            print(f"⚠️ Error pointer failed: {e}")
            return frame
    
    def _add_text_overlay(self, frame: np.ndarray, feedback: str, problem_location: str) -> np.ndarray:
        """Add text overlay to analysis frame"""
        try:
            # Add background rectangle
            cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, 200), (0, 0, 0, 180), -1)
            
            # Add text
            cv2.putText(frame, f"Problem: {problem_location.upper()}", (70, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Wrap feedback text
            words = feedback.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 60:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
            
            # Add wrapped text
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                cv2.putText(frame, line, (70, 130 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Text overlay failed: {e}")
            return frame
    
    def _create_fallback_frame(self, feedback: str) -> str:
        """Create fallback frame when MediaPipe fails"""
        try:
            # Create simple black frame with text
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Add text
            cv2.putText(frame, "ANALYSIS FRAME", (800, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, feedback[:50] + "...", (600, 500), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            output_frame = "temp/fallback_frame.jpg"
            cv2.imwrite(output_frame, frame)
            return output_frame
            
        except Exception as e:
            print(f"⚠️ Fallback frame failed: {e}")
            return ""
    
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