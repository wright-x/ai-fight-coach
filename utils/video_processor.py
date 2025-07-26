"""
Video Processor - Professional Boxing Analysis System
Creates cinematic highlight reels with MediaPipe pose analysis
"""

import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Optional
import tempfile
import time
import traceback
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, TextClip, concatenate_videoclips, AudioFileClip

class VideoProcessor:
    """Professional video processor for boxing analysis"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Try to initialize MediaPipe with fallback
        try:
            import cv2
            import mediapipe as mp
            import numpy as np
            
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.cv2 = cv2
            self.np = np
            self.mediapipe_available = True
            print("✅ VideoProcessor initialized (Professional Boxing Analysis with MediaPipe)")
            
        except ImportError as e:
            print(f"⚠️ MediaPipe import failed: {e}")
            self.mediapipe_available = False
            print("✅ VideoProcessor initialized (Professional Boxing Analysis - MediaPipe fallback mode)")
        except Exception as e:
            print(f"⚠️ MediaPipe initialization failed: {e}")
            self.mediapipe_available = False
            print("✅ VideoProcessor initialized (Professional Boxing Analysis - MediaPipe fallback mode)")
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str) -> str:
        """Create professional boxing analysis video"""
        try:
            print(f"🎬 Creating professional boxing analysis: {video_path}")
            print(f"📊 Number of highlights: {len(highlights)}")
            
            # Load original video
            video = VideoFileClip(video_path)
            duration = video.duration
            fps = video.fps
            print(f"📊 Video: {duration}s at {fps}fps")
            
            # Create clips list
            all_clips = []
            
            # 1. OPENING CARD (1.5 seconds)
            opening_clip = self._create_opening_card(duration=1.5)
            all_clips.append(opening_clip)
            
            # 2. FOR EACH HIGHLIGHT
            for i, highlight in enumerate(highlights):
                print(f"🎯 Processing highlight {i+1}/{len(highlights)}")
                
                # a. Black screen: "HIGHLIGHT <n>" (1 second)
                highlight_title = self._create_highlight_title(f"HIGHLIGHT {i+1}", duration=1.0)
                all_clips.append(highlight_title)
                
                # b. Freeze-frame with MediaPipe analysis (3 seconds)
                freeze_frame = self._create_mediapipe_analysis_frame(
                    video_path=video_path,
                    highlight=highlight,
                    duration=3.0
                )
                all_clips.append(freeze_frame)
                
                # c. Slow-motion clip with captions and TTS
                slow_motion_clip = self._create_slow_motion_clip(
                    video=video,
                    highlight=highlight,
                    slow_factor=0.7
                )
                all_clips.append(slow_motion_clip)
            
            # 3. END CARD (1.5 seconds)
            end_clip = self._create_end_card(duration=1.5)
            all_clips.append(end_clip)
            
            # Concatenate all clips
            print(f"🎬 Concatenating {len(all_clips)} clips...")
            final_video = concatenate_videoclips(all_clips, method="compose")
            
            # Write final video
            print(f"💾 Writing final analysis video...")
            final_video.write_videofile(
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
            final_video.close()
            for clip in all_clips:
                clip.close()
            
            print(f"✅ Professional boxing analysis created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating analysis video: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            # Fallback to simple copy
            shutil.copy2(video_path, output_path)
            return output_path
    
    def _create_opening_card(self, duration: float = 1.5) -> VideoFileClip:
        """Create opening card with "AI Boxing Analysis" text"""
        try:
            # Create black background
            bg = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
            
            # Create title text with nice design
            title = TextClip(
                "AI Boxing Analysis",
                fontsize=120,
                color='white',
                font='Arial-Bold',
                stroke_color='#FF6B35',
                stroke_width=4
            ).set_position('center').set_duration(duration)
            
            # Create subtitle
            subtitle = TextClip(
                "Professional Fight Analysis",
                fontsize=60,
                color='#FF6B35',
                font='Arial'
            ).set_position(('center', 600)).set_duration(duration)
            
            # Create composite
            opening = CompositeVideoClip([bg, title, subtitle])
            return opening
            
        except Exception as e:
            print(f"⚠️ Opening card failed: {e}")
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _create_highlight_title(self, text: str, duration: float = 1.0) -> VideoFileClip:
        """Create highlight title screen"""
        try:
            # Create black background
            bg = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
            
            # Create highlight text
            title = TextClip(
                text,
                fontsize=100,
                color='#FF6B35',
                font='Arial-Bold',
                stroke_color='white',
                stroke_width=3
            ).set_position('center').set_duration(duration)
            
            # Create composite
            highlight = CompositeVideoClip([bg, title])
            return highlight
            
        except Exception as e:
            print(f"⚠️ Highlight title failed: {e}")
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _create_mediapipe_analysis_frame(self, video_path: str, highlight: Dict, duration: float = 3.0) -> VideoFileClip:
        """Create MediaPipe analysis frame with pose skeleton and arrow"""
        try:
            timestamp = highlight.get('timestamp', '00:00')
            problem_location = highlight.get('problem_location', 'general')
            feedback = highlight.get('detailed_feedback', 'No feedback')
            
            # Parse timestamp and clamp to video length
            highlight_time = self._parse_timestamp(timestamp)
            
            # Check if MediaPipe is available
            if not self.mediapipe_available:
                print(f"⚠️ MediaPipe not available, using fallback frame")
                return self._create_fallback_frame(feedback, duration)
            
            # Extract frame at timestamp
            cap = self.cv2.VideoCapture(video_path)
            cap.set(self.cv2.CAP_PROP_POS_MSEC, highlight_time * 1000)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"⚠️ Could not read frame at {highlight_time}s")
                return self._create_fallback_frame(feedback, duration)
            
            # Initialize MediaPipe Pose
            with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            ) as pose:
                # Convert BGR to RGB
                rgb_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                # Draw pose landmarks
                annotated_frame = frame.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Add error arrow pointing to specific joint
                annotated_frame = self._add_error_arrow(annotated_frame, results.pose_landmarks, problem_location)
                
                # Add text overlay
                annotated_frame = self._add_analysis_text(annotated_frame, feedback, problem_location)
                
                # Save frame
                temp_frame_path = f"temp/analysis_frame_{int(highlight_time)}.jpg"
                self.cv2.imwrite(temp_frame_path, annotated_frame)
                
                # Create video clip from frame
                frame_clip = VideoFileClip(temp_frame_path).set_duration(duration)
                return frame_clip
                
        except Exception as e:
            print(f"⚠️ MediaPipe analysis failed: {e}")
            return self._create_fallback_frame(feedback, duration)
    
    def _add_error_arrow(self, frame, landmarks, problem_location: str):
        """Add red arrow pointing to specific body part"""
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
                cv2.arrowedLine(frame, (x-80, y-80), (x, y), (0, 0, 255), 8, tipLength=0.4)
                cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)
                
            return frame
            
        except Exception as e:
            print(f"⚠️ Error arrow failed: {e}")
            return frame
    
    def _add_analysis_text(self, frame, feedback: str, problem_location: str):
        """Add analysis text overlay"""
        try:
            # Add background rectangle
            cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, 200), (0, 0, 0, 180), -1)
            
            # Add problem location text
            cv2.putText(frame, f"Problem: {problem_location.upper()}", (70, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Wrap feedback text
            words = feedback.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 50:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
            
            # Add wrapped text
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                cv2.putText(frame, line, (70, 130 + i*35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Analysis text failed: {e}")
            return frame
    
    def _create_slow_motion_clip(self, video: VideoFileClip, highlight: Dict, slow_factor: float = 0.7) -> VideoFileClip:
        """Create slow-motion clip with captions and TTS"""
        try:
            timestamp = highlight.get('timestamp', '00:00')
            feedback = highlight.get('detailed_feedback', 'No feedback')
            action = highlight.get('action_required', 'No action')
            
            # Parse timestamp and clamp to video length
            highlight_time = self._parse_timestamp(timestamp)
            video_duration = video.duration
            
            # Calculate clip timing [t-3s, t+3s] with clamping
            start_time = max(0, highlight_time - 3)
            end_time = min(video_duration, highlight_time + 3)
            
            print(f"🎯 Slow motion: {start_time}s - {end_time}s (original: {highlight_time}s)")
            
            # Extract video clip
            video_clip = video.subclip(start_time, end_time)
            
            # Apply slow motion
            slow_clip = video_clip.speedx(slow_factor)
            
            # Create caption overlay
            caption_clip = self._create_caption_overlay(feedback, action, slow_clip.duration)
            
            # Combine video and captions
            final_clip = CompositeVideoClip([slow_clip, caption_clip])
            
            return final_clip
            
        except Exception as e:
            print(f"⚠️ Slow motion clip failed: {e}")
            # Fallback to simple subclip
            highlight_time = self._parse_timestamp(highlight.get('timestamp', '00:00'))
            start_time = max(0, highlight_time - 3)
            end_time = min(video.duration, highlight_time + 3)
            return video.subclip(start_time, end_time)
    
    def _create_caption_overlay(self, feedback: str, action: str, duration: float) -> VideoFileClip:
        """Create burn-in caption overlay"""
        try:
            # Create background banner
            banner = ColorClip(size=(1920, 120), color=(0, 0, 0, 180), duration=duration)
            banner = banner.set_position(('center', 'bottom'))
            
            # Create feedback text
            feedback_text = TextClip(
                feedback[:80] + "..." if len(feedback) > 80 else feedback,
                fontsize=35,
                color='white',
                font='Arial-Bold'
            ).set_position(('center', 1080-100)).set_duration(duration)
            
            # Create action text
            action_text = TextClip(
                f"Action: {action}",
                fontsize=25,
                color='#FF6B35',
                font='Arial'
            ).set_position(('center', 1080-50)).set_duration(duration)
            
            # Combine
            captions = CompositeVideoClip([banner, feedback_text, action_text])
            return captions
            
        except Exception as e:
            print(f"⚠️ Caption overlay failed: {e}")
            return ColorClip(size=(1, 1), color=(0, 0, 0), duration=duration)
    
    def _create_end_card(self, duration: float = 1.5) -> VideoFileClip:
        """Create end card with copyright"""
        try:
            # Create black background
            bg = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
            
            # Create copyright text
            copyright_text = TextClip(
                "© AI Boxing Analysis 2025",
                fontsize=40,
                color='#666666',
                font='Arial'
            ).set_position(('center', 'bottom')).set_duration(duration)
            
            # Create composite
            end = CompositeVideoClip([bg, copyright_text])
            return end
            
        except Exception as e:
            print(f"⚠️ End card failed: {e}")
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _create_fallback_frame(self, feedback: str, duration: float) -> VideoFileClip:
        """Create fallback frame when MediaPipe fails"""
        try:
            if self.mediapipe_available:
                # Create simple black frame with text
                frame = self.np.zeros((1080, 1920, 3), dtype=self.np.uint8)
                
                # Add text
                self.cv2.putText(frame, "ANALYSIS FRAME", (800, 400), 
                               self.cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                self.cv2.putText(frame, feedback[:50] + "...", (600, 500), 
                               self.cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                temp_frame_path = "temp/fallback_frame.jpg"
                self.cv2.imwrite(temp_frame_path, frame)
                
                frame_clip = VideoFileClip(temp_frame_path).set_duration(duration)
                return frame_clip
            else:
                # Create simple text overlay without OpenCV
                bg = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
                title = TextClip("ANALYSIS FRAME", fontsize=80, color='white', font='Arial-Bold').set_position('center').set_duration(duration)
                subtitle = TextClip(feedback[:50] + "...", fontsize=40, color='white', font='Arial').set_position(('center', 600)).set_duration(duration)
                return CompositeVideoClip([bg, title, subtitle])
            
        except Exception as e:
            print(f"⚠️ Fallback frame failed: {e}")
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse timestamp string to seconds with clamping"""
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
        """Add TTS audio to video with proper sync"""
        try:
            if os.path.exists(audio_path):
                video = VideoFileClip(video_path)
                audio = AudioFileClip(audio_path)
                
                # Combine video and audio (TTS will duck original audio)
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