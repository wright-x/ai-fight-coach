"""
AI Boxing Analysis Video Processor - Fallback Version
Simplified version without MediaPipe for deployment compatibility
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import time
import traceback
import shutil

class VideoProcessorFallback:
    """Simplified video processor for boxing analysis without MediaPipe"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Color scheme for sentiment-based coloring
        self.colors = {
            'problem': (255, 65, 54),      # RED #FF4136
            'action': (46, 204, 64),       # GREEN #2ECC40
            'neutral': (255, 255, 255),    # WHITE #FFFFFF
            'outline': (0, 0, 0),          # BLACK for text outlines
            'head_pointer': (0, 123, 255)  # Blue head pointer
        }
        
        print("✅ VideoProcessorFallback initialized (no MediaPipe)")
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str, user_name: str = "FIGHTER") -> str:
        """Create simplified boxing analysis video"""
        try:
            print(f"🎬 Creating simplified boxing analysis: {video_path}")
            print(f"📊 Number of highlights: {len(highlights)}")
            print(f"👤 User: {user_name}")
            
            # Load and analyze video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps
            
            print(f"📊 Video: {width}x{height}, {fps}fps, {duration:.2f}s")
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Generate video segments
            all_segments = []
            
            # 1. Opening Card (1.5 seconds)
            print("🎬 Creating opening card...")
            opening_frames = self._create_opening_card(width, height, fps, 1.5)
            all_segments.extend(opening_frames)
            
            # 2. Process each highlight
            for i, highlight in enumerate(highlights):
                print(f"🎯 Processing highlight {i+1}/{len(highlights)}")
                
                # Title card (1 second)
                title_frames = self._create_title_card(f"HIGHLIGHT {i+1}", width, height, fps, 1.0)
                all_segments.extend(title_frames)
                
                # Analysis frame (3 seconds freeze-frame)
                analysis_frames = self._create_analysis_frame(
                    cap, highlight, width, height, fps, 3.0, user_name
                )
                all_segments.extend(analysis_frames)
                
                # Slow motion clip (6 seconds at 0.5x speed)
                slow_motion_frames = self._create_slow_motion_clip(
                    cap, highlight, width, height, fps, 6.0, user_name
                )
                all_segments.extend(slow_motion_frames)
            
            # 3. End Card (1.5 seconds)
            print("🎬 Creating end card...")
            end_frames = self._create_end_card(width, height, fps, 1.5)
            all_segments.extend(end_frames)
            
            # Write all frames to video
            print(f"💾 Writing {len(all_segments)} frames to video...")
            for frame in all_segments:
                out.write(frame)
            
            # Cleanup
            cap.release()
            out.release()
            
            print(f"✅ Simplified boxing analysis created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating analysis video: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            # Fallback to simple copy
            shutil.copy2(video_path, output_path)
            return output_path
    
    def _create_opening_card(self, width: int, height: int, fps: float, duration: float) -> List[np.ndarray]:
        """Create opening card with title and subtitle"""
        frames = []
        num_frames = int(fps * duration)
        
        for _ in range(num_frames):
            # Create black background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add title text
            title = "AI Boxing Analysis"
            subtitle = "Professional Fight Analysis"
            
            # Calculate text positions (centered)
            title_y = height // 2 - 50
            subtitle_y = height // 2 + 50
            
            # Draw text with outline
            self._draw_text_with_outline(frame, title, width//2, title_y, 
                                       self.colors['neutral'], 2.0, 4)
            self._draw_text_with_outline(frame, subtitle, width//2, subtitle_y, 
                                       self.colors['action'], 1.0, 2)
            
            frames.append(frame)
        
        return frames
    
    def _create_title_card(self, title: str, width: int, height: int, fps: float, duration: float) -> List[np.ndarray]:
        """Create title card for each highlight"""
        frames = []
        num_frames = int(fps * duration)
        
        for _ in range(num_frames):
            # Create black background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw title text
            self._draw_text_with_outline(frame, title, width//2, height//2, 
                                       self.colors['action'], 2.5, 6)
            
            frames.append(frame)
        
        return frames
    
    def _create_analysis_frame(self, cap: cv2.VideoCapture, highlight: Dict, 
                             width: int, height: int, fps: float, duration: float, 
                             user_name: str) -> List[np.ndarray]:
        """Create analysis frame without MediaPipe"""
        frames = []
        num_frames = int(fps * duration)
        
        # Get timestamp and extract frame
        timestamp = self._parse_timestamp(highlight.get('timestamp', '00:00'))
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️ Could not read frame at {timestamp}s")
            # Create fallback frame
            for _ in range(num_frames):
                fallback_frame = self._create_fallback_frame(width, height, highlight)
                frames.append(fallback_frame)
            return frames
        
        # Add simple head pointer (no MediaPipe)
        frame = self._add_simple_head_pointer(frame, user_name)
        
        # Add text overlays
        frame = self._add_analysis_text_overlay(frame, highlight)
        
        # Repeat frame for duration
        for _ in range(num_frames):
            frames.append(frame.copy())
        
        return frames
    
    def _add_simple_head_pointer(self, frame: np.ndarray, user_name: str) -> np.ndarray:
        """Add simple head pointer without MediaPipe"""
        try:
            h, w, _ = frame.shape
            
            # Simple head detection (assume center-top area)
            head_x = w // 2
            head_y = h // 4
            
            # Draw pointer above head
            pointer_y = max(50, head_y - 80)
            
            # Draw arrow pointing down to head
            cv2.arrowedLine(frame, (head_x, pointer_y), (head_x, head_y-20), 
                           self.colors['head_pointer'], 4, tipLength=0.3)
            
            # Draw circle around head
            cv2.circle(frame, (head_x, head_y), 25, self.colors['head_pointer'], 3)
            
            # Add user name above pointer
            self._draw_text_with_outline(frame, user_name, head_x, pointer_y-20, 
                                       self.colors['head_pointer'], 0.8, 2)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Simple head pointer failed: {e}")
            return frame
    
    def _add_analysis_text_overlay(self, frame: np.ndarray, highlight: Dict) -> np.ndarray:
        """Add analysis text overlay with sentiment-based coloring"""
        try:
            h, w, _ = frame.shape
            
            # Create semi-transparent background at bottom
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-200), (w, h), (0, 0, 0, 180), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Get text content
            problem_location = highlight.get('problem_location', 'general')
            feedback = highlight.get('detailed_feedback', 'No feedback')
            
            # Add problem location text
            problem_text = f"PROBLEM: {problem_location.upper()}"
            self._draw_text_with_outline(frame, problem_text, w//2, h-160, 
                                       self.colors['neutral'], 1.2, 3)
            
            # Add feedback text with sentiment coloring
            self._draw_sentiment_text(frame, feedback, w//2, h-120, w)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Analysis text overlay failed: {e}")
            return frame
    
    def _draw_sentiment_text(self, frame: np.ndarray, text: str, x: int, y: int, width: int):
        """Draw text with sentiment-based coloring"""
        # Simple sentiment analysis
        problem_keywords = ['problem', 'error', 'wrong', 'bad', 'poor', 'weak', 'flat', 'linear']
        action_keywords = ['should', 'must', 'need', 'improve', 'better', 'correct', 'fix', 'maintain']
        
        words = text.split()
        current_line = ""
        line_y = y
        max_width = width - 100  # Leave margins
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            
            # Determine color based on sentiment
            if any(keyword in word.lower() for keyword in problem_keywords):
                color = self.colors['problem']
            elif any(keyword in word.lower() for keyword in action_keywords):
                color = self.colors['action']
            else:
                color = self.colors['neutral']
            
            # Check if line would be too long
            if len(test_line) * 20 > max_width:  # Rough estimate
                if current_line:
                    self._draw_text_with_outline(frame, current_line, x, line_y, 
                                               self.colors['neutral'], 0.8, 2)
                    line_y += 25
                    current_line = word
                else:
                    # Word is too long, break it
                    self._draw_text_with_outline(frame, word, x, line_y, color, 0.8, 2)
                    line_y += 25
            else:
                current_line = test_line
        
        # Draw remaining text
        if current_line:
            self._draw_text_with_outline(frame, current_line, x, line_y, 
                                       self.colors['neutral'], 0.8, 2)
    
    def _create_slow_motion_clip(self, cap: cv2.VideoCapture, highlight: Dict, 
                               width: int, height: int, fps: float, duration: float, 
                               user_name: str) -> List[np.ndarray]:
        """Create slow motion clip with captions"""
        frames = []
        
        # Calculate timing
        timestamp = self._parse_timestamp(highlight.get('timestamp', '00:00'))
        start_time = max(0, timestamp - 3)
        end_time = timestamp + 3
        
        # Extract frames at 0.5x speed
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        frame_count = 0
        target_frames = int(fps * duration)
        
        while frame_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add simple head pointer
            frame = self._add_simple_head_pointer(frame, user_name)
            
            # Add captions
            frame = self._add_slow_motion_captions(frame, highlight)
            
            frames.append(frame)
            frame_count += 1
            
            # Skip every other frame for 0.5x speed effect
            cap.read()  # Skip frame
        
        return frames
    
    def _add_slow_motion_captions(self, frame: np.ndarray, highlight: Dict) -> np.ndarray:
        """Add captions during slow motion"""
        try:
            h, w, _ = frame.shape
            
            # Create semi-transparent background at bottom
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-150), (w, h), (0, 0, 0, 180), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Add action text
            action = highlight.get('action_required', 'No action specified')
            self._draw_text_with_outline(frame, f"ACTION: {action}", w//2, h-100, 
                                       self.colors['action'], 1.0, 3)
            
            # Add TTS indicator
            self._draw_text_with_outline(frame, "🎤 TTS NARRATION", w//2, h-50, 
                                       self.colors['action'], 0.8, 2)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Slow motion captions failed: {e}")
            return frame
    
    def _create_end_card(self, width: int, height: int, fps: float, duration: float) -> List[np.ndarray]:
        """Create end card with copyright"""
        frames = []
        num_frames = int(fps * duration)
        
        for _ in range(num_frames):
            # Create black background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add copyright text
            copyright_text = "© AI Boxing Analysis 2025"
            self._draw_text_with_outline(frame, copyright_text, width//2, height//2, 
                                       self.colors['neutral'], 1.0, 2)
            
            frames.append(frame)
        
        return frames
    
    def _create_fallback_frame(self, width: int, height: int, highlight: Dict) -> np.ndarray:
        """Create fallback frame when analysis fails"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add simple text
        feedback = highlight.get('detailed_feedback', 'Analysis frame')
        self._draw_text_with_outline(frame, "ANALYSIS FRAME", width//2, height//2-50, 
                                   self.colors['neutral'], 1.5, 3)
        self._draw_text_with_outline(frame, feedback[:50] + "...", width//2, height//2+50, 
                                   self.colors['neutral'], 1.0, 2)
        
        return frame
    
    def _draw_text_with_outline(self, frame: np.ndarray, text: str, x: int, y: int, 
                               color: Tuple[int, int, int], scale: float, thickness: int):
        """Draw text with black outline for readability"""
        # Draw black outline
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 
                   self.colors['outline'], thickness + 2)
        
        # Draw colored text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 
                   color, thickness)
    
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
        """Add TTS audio to video with proper sync"""
        try:
            if os.path.exists(audio_path):
                # Use ffmpeg to combine video and audio
                import subprocess
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-shortest',
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"✅ Audio added to video: {output_path}")
                return output_path
            else:
                print(f"⚠️ Audio file not found: {audio_path}")
                shutil.copy2(video_path, output_path)
                return output_path
                
        except Exception as e:
            print(f"❌ Error adding audio to video: {e}")
            shutil.copy2(video_path, output_path)
            return output_path 