"""
AI Boxing Analysis Video Processor - Simplified Version
Inspired by gemini-bball repository for clean, reliable video processing
"""

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Any

class VideoProcessor:
    """Simplified video processor for boxing analysis with clean visual style"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        print("✅ VideoProcessor initialized (simplified mode)")
        
        # Colors inspired by gemini-bball
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255)
        }
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str, user_name: str = "FIGHTER") -> str:
        """Create boxing analysis video with simplified workflow"""
        try:
            print(f"🎬 Creating boxing analysis: {video_path}")
            print(f"📊 Highlights: {len(highlights)}")
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📊 Video: {width}x{height}, {fps}fps")
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise Exception(f"Could not create output video: {output_path}")
            
            # Process each highlight
            for i, highlight in enumerate(highlights):
                print(f"🎬 Processing highlight {i+1}/{len(highlights)}")
                
                # Create highlight clip with slow motion and captions
                highlight_frames = self._create_highlight_clip(
                    cap, highlight, width, height, fps, user_name
                )
                
                # Write frames to output
                for frame in highlight_frames:
                    out.write(frame)
            
            # Cleanup
            cap.release()
            out.release()
            
            print(f"✅ Analysis video created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating analysis video: {e}")
            # Fallback: copy original video
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path
    
    def _create_highlight_clip(self, cap: cv2.VideoCapture, highlight: Dict, 
                             width: int, height: int, fps: float, user_name: str) -> List[np.ndarray]:
        """Create a single highlight clip with slow motion and captions"""
        frames = []
        
        # Parse timestamp
        timestamp = self._parse_timestamp(highlight.get('timestamp', '00:15'))
        frame_number = int(timestamp * fps)
        
        # Set video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - int(fps * 1.5)))  # 1.5s before
        
        # Read frames for 3 seconds (slow motion effect)
        frame_count = 0
        max_frames = int(fps * 3)  # 3 seconds
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Slow motion: repeat each frame 2 times (0.5x speed)
            for _ in range(2):
                # Add head pointer
                frame = self._add_head_pointer(frame, user_name)
                
                # Add animated captions
                frame = self._add_animated_captions(frame, highlight, frame_count, fps)
                
                frames.append(frame)
            
            frame_count += 1
        
        return frames
    
    def _add_head_pointer(self, frame: np.ndarray, user_name: str) -> np.ndarray:
        """Add simple head pointer (circle + line + label) inspired by gemini-bball"""
        height, width = frame.shape[:2]
        
        # Estimate head position (center-top of frame)
        head_x = width // 2
        head_y = height // 4
        
        # Draw circle around head
        cv2.circle(frame, (head_x, head_y), 20, self.colors['blue'], 3)
        
        # Draw line to label
        label_x = head_x + 30
        label_y = head_y - 30
        
        cv2.line(frame, (head_x, head_y), (label_x, label_y), self.colors['blue'], 2)
        
        # Draw label background
        label_text = user_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(frame, 
                     (label_x - 5, label_y - text_height - 5),
                     (label_x + text_width + 5, label_y + baseline + 5),
                     self.colors['black'], -1)
        
        # Text
        cv2.putText(frame, label_text, (label_x, label_y), 
                   font, font_scale, self.colors['white'], thickness)
        
        return frame
    
    def _add_animated_captions(self, frame: np.ndarray, highlight: Dict, 
                             frame_count: int, fps: float) -> np.ndarray:
        """Add animated captions that appear word by word"""
        height, width = frame.shape[:2]
        
        # Get caption text
        feedback = highlight.get('detailed_feedback', 'Good technique observed')
        action = highlight.get('action_required', 'Continue practicing')
        
        # Calculate animation timing
        words_per_second = 2  # Words appear at 2 per second
        total_duration = len(feedback.split()) / words_per_second
        current_time = frame_count / fps
        
        # Determine which words to show
        words = feedback.split()
        words_to_show = min(len(words), int(current_time * words_per_second))
        
        # Build caption text
        caption_text = ' '.join(words[:words_to_show])
        
        # Draw caption background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(caption_text, font, font_scale, thickness)
        
        # Position at bottom center
        text_x = (width - text_width) // 2
        text_y = height - 100
        
        # Background rectangle
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_height - 10),
                     (text_x + text_width + 10, text_y + baseline + 10),
                     self.colors['black'], -1)
        
        # Caption text
        cv2.putText(frame, caption_text, (text_x, text_y), 
                   font, font_scale, self.colors['white'], thickness)
        
        # Action text (always visible)
        action_x = (width - cv2.getTextSize(action, font, font_scale, thickness)[0][0]) // 2
        action_y = text_y + 40
        
        cv2.putText(frame, action, (action_x, action_y), 
                   font, font_scale, self.colors['yellow'], thickness)
        
        return frame
    
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
            return 15.0  # Default to 15 seconds
    
    def add_audio_to_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Add audio to video using moviepy"""
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            print(f"🎬 Adding audio to video...")
            
            # Load clips
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Combine
            final_clip = video_clip.set_audio(audio_clip)
            
            # Write output
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False)
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            print(f"✅ Audio added successfully")
            return output_path
            
        except Exception as e:
            print(f"❌ Error adding audio: {e}")
            # Fallback: copy video without audio
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path 