"""
Video Processor - Railway Compatible Version
Handles video editing, overlays, and head tracking
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from typing import List, Dict, Any, Optional
import tempfile
import time

class VideoProcessor:
    """Video processor for Railway environment with head tracking and overlays"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ VideoProcessor initialized (full mode)")
    
    def process_video(self, video_path: str, fighter_name: str = "FIGHTER") -> Dict[str, Any]:
        """Process video and return analysis results"""
        try:
            # This will be handled by Gemini client
            return {"status": "ready_for_analysis"}
        except Exception as e:
            print(f"Error processing video: {e}")
            return {"status": "error", "message": str(e)}
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str) -> str:
        """Create highlight video with overlays and head tracking"""
        try:
            print(f"🎬 Creating highlight video: {video_path}")
            
            # Load video
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            current_highlight = None
            highlight_index = 0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                current_time = frame_count / fps
                
                # Check if we're in a highlight moment
                if highlight_index < len(highlights):
                    highlight = highlights[highlight_index]
                    highlight_time = self._parse_timestamp(highlight.get('timestamp', '0'))
                    
                    if current_time >= highlight_time and current_highlight != highlight:
                        current_highlight = highlight
                        highlight_index += 1
                
                # Process frame with head tracking
                processed_frame = self._process_frame_with_overlays(
                    frame, current_highlight, fighter_name, current_time
                )
                
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"📹 Processed {frame_count} frames...")
            
            video.release()
            out.release()
            
            print(f"✅ Highlight video created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating highlight video: {e}")
            return video_path
    
    def _process_frame_with_overlays(self, frame: np.ndarray, highlight: Optional[Dict], 
                                   fighter_name: str, current_time: float) -> np.ndarray:
        """Process frame with head tracking and overlays"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Draw head tracking
            if results.pose_landmarks:
                # Get nose position (head center)
                nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                h, w, _ = frame.shape
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)
                
                # Draw fighter name above head
                cv2.putText(frame, fighter_name, (nose_x - 50, nose_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw head tracking circle
                cv2.circle(frame, (nose_x, nose_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (nose_x, nose_y), 15, (0, 255, 0), 2)
            
            # Add highlight overlay if we're in a highlight moment
            if highlight:
                self._add_highlight_overlay(frame, highlight)
            
            return frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def _add_highlight_overlay(self, frame: np.ndarray, highlight: Dict):
        """Add highlight text overlay to frame"""
        try:
            h, w, _ = frame.shape
            
            # Create semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add highlight text
            feedback = highlight.get('detailed_feedback', '')
            action = highlight.get('action_required', '')
            
            # Split long text into lines
            words = feedback.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) < 60:
                    current_line += " " + word if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw text lines
            y_offset = 30
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                cv2.putText(frame, line, (20, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add action required
            if action:
                cv2.putText(frame, f"Action: {action}", (20, y_offset + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error adding highlight overlay: {e}")
    
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