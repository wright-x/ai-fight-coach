import os
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont
import base64
import tempfile

class VideoProcessor:
    """Surgical VideoProcessor that follows exact specifications"""
    
    def __init__(self):
        # Initialize MediaPipe for head tracking
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Colors for overlays
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'magenta': (255, 0, 255)  # For head pointer
        }
        
        print("✅ VideoProcessor initialized (surgical mode)")

    def create_highlight_video(self, video_path: str, highlights: list, output_path: str, user_name: str = "FIGHTER") -> str:
        """
        Main entry point - creates final video with t-3 to t+3 highlights at 0.7x speed
        """
        try:
            print(f"🎬 Starting surgical video processing for {len(highlights)} highlights")
            
            # Load source video
            source_clip = VideoFileClip(video_path)
            source_fps = source_clip.fps
            source_duration = source_clip.duration
            
            print(f"📹 Source video: {source_duration:.2f}s at {source_fps}fps")
            
            processed_clips = []
            
            # Process each highlight with surgical precision
            for i, highlight in enumerate(highlights):
                print(f"🎯 Processing highlight {i+1}/{len(highlights)}")
                
                # Extract timestamp and convert to seconds
                timestamp = self._parse_timestamp(highlight.get('timestamp', '00:00'))
                
                # CRITICAL: t-3 to t+3 window
                start_time = max(0, timestamp - 3)
                end_time = min(source_duration, timestamp + 3)
                
                print(f"⏰ Highlight window: {start_time:.2f}s to {end_time:.2f}s")
                
                # Extract the highlight clip
                highlight_clip = source_clip.subclip(start_time, end_time)
                
                # CRITICAL: Slow down to 0.7x speed
                slowed_clip = highlight_clip.speedx(0.7)
                
                # Process frames with overlays
                processed_frames = []
                for frame in slowed_clip.iter_frames():
                    processed_frame = self._add_overlays(
                        frame, 
                        highlight.get('detailed_feedback', ''),
                        highlight.get('action_required', ''),
                        user_name
                    )
                    processed_frames.append(processed_frame)
                
                # Create new clip from processed frames
                processed_clip = ImageSequenceClip(processed_frames, fps=slowed_clip.fps)
                
                # Add TTS audio if available
                if 'tts_audio' in highlight:
                    try:
                        audio_clip = AudioFileClip(highlight['tts_audio'])
                        processed_clip = processed_clip.set_audio(audio_clip)
                    except Exception as e:
                        print(f"⚠️ TTS audio failed for highlight {i+1}: {e}")
                
                processed_clips.append(processed_clip)
                
                # Cleanup
                highlight_clip.close()
                slowed_clip.close()
            
            # CRITICAL: Concatenate all clips
            if processed_clips:
                final_video = concatenate_videoclips(processed_clips)
                
                # Write final video with proper codec
                final_video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=source_fps,
                    verbose=False,
                    logger=None
                )
                
                # Cleanup
                final_video.close()
                for clip in processed_clips:
                    clip.close()
                
                print(f"✅ Surgical video processing complete: {output_path}")
                return output_path
            else:
                raise ValueError("No valid clips were processed")
                
        except Exception as e:
            print(f"❌ Surgical video processing failed: {e}")
            # Fallback: copy original video
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path
        finally:
            # Cleanup MediaPipe
            if hasattr(self, 'mp_pose'):
                self.mp_pose.close()
            if 'source_clip' in locals():
                source_clip.close()

    def _add_overlays(self, frame, feedback_text, action_text, user_name):
        """
        Add head pointer and static captions with thick black outline
        """
        try:
            # Convert frame to PIL for text rendering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_frame)
            
            h, w, _ = frame.shape
            
            # CRITICAL: Head pointer using MediaPipe
            head_pos = self._detect_head_position(frame)
            if head_pos:
                x, y = head_pos
                # Draw magenta circle pointer
                radius = max(10, int(w * 0.02))  # 2% of frame width, minimum 10px
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    outline=self.colors['magenta'],
                    width=4
                )
                
                # Add "FIGHTER" label
                label_text = user_name
                label_font_size = max(20, int(w * 0.025))
                try:
                    label_font = ImageFont.truetype("arial.ttf", label_font_size)
                except:
                    label_font = ImageFont.load_default()
                
                # Position label above the pointer
                label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                label_width = label_bbox[2] - label_bbox[0]
                label_x = x - label_width // 2
                label_y = y - radius - label_font_size - 5
                
                # Draw label with outline
                draw.text(
                    (label_x, label_y),
                    label_text,
                    font=label_font,
                    fill=self.colors['white'],
                    stroke_width=2,
                    stroke_fill=self.colors['black']
                )
            
            # CRITICAL: Static captions at bottom center
            if feedback_text or action_text:
                # Combine text
                caption_text = f"{feedback_text}\n{action_text}" if action_text else feedback_text
                
                # Calculate font size (4% of frame width)
                font_size = max(24, int(w * 0.04))
                
                try:
                    # Try to use a system font
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # CRITICAL: Position at bottom center with 5% margin
                text_bbox = draw.textbbox((0, 0), caption_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = (w - text_width) // 2
                text_y = h - text_height - int(h * 0.05)  # 5% margin from bottom
                
                # CRITICAL: Draw text with thick black outline (no background)
                draw.text(
                    (text_x, text_y),
                    caption_text,
                    font=font,
                    fill=self.colors['white'],
                    stroke_width=4,  # Thick outline
                    stroke_fill=self.colors['black']
                )
            
            # Convert back to numpy array
            return np.array(pil_frame)
            
        except Exception as e:
            print(f"⚠️ Overlay rendering failed: {e}")
            return frame

    def _detect_head_position(self, frame):
        """
        Detect head position using MediaPipe pose landmarks
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Use nose landmark for head position
                nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                h, w, _ = frame.shape
                x = int(nose.x * w)
                y = int(nose.y * h)
                return (x, y)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Head detection failed: {e}")
            return None

    def _parse_timestamp(self, timestamp_str):
        """
        Convert timestamp string (MM:SS) to seconds
        """
        try:
            if ':' in timestamp_str:
                parts = timestamp_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = map(int, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
            return 0
        except:
            return 0

    def add_audio_to_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        Combine video with TTS audio using MoviePy
        """
        try:
            print(f"🔊 Adding audio to video: {audio_path}")
            
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Set audio to video
            final_clip = video_clip.set_audio(audio_clip)
            
            # Write final video
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None
            )
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            print(f"✅ Audio added successfully: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Audio addition failed: {e}")
            # Fallback: copy video without audio
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path 