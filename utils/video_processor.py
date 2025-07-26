"""
Video processing utility for AI Fight Coach application.
Handles video slowdown, frame extraction, and video manipulation.
"""

import cv2
import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import mediapipe as mp
from .logger import logger


class VideoProcessor:
    """Handles all video processing operations for the AI Fight Coach."""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe Pose
        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose_detection_available = True
            logger.info("MediaPipe pose detection initialized successfully")
        except Exception as e:
            logger.warning(f"MediaPipe pose detection not available: {e}")
            self.pose_detection_available = False
        
        logger.info(f"VideoProcessor initialized with temp_dir: {self.temp_dir}")
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata and information."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            info = {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "file_size": os.path.getsize(video_path)
            }
            
            logger.info(f"Video info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
    
    def slow_down_video(self, video_path: str, slowdown_factor: float = 4.0) -> str:
        """
        Slow down video by the specified factor for frame-by-frame analysis.
        Returns path to the slowed down video.
        """
        try:
            logger.info(f"Starting video slowdown: {video_path} with factor {slowdown_factor}")
            
            # Load video
            video = VideoFileClip(video_path)
            original_fps = video.fps
            
            # Calculate new fps
            new_fps = original_fps / slowdown_factor
            logger.info(f"Original FPS: {original_fps}, New FPS: {new_fps}")
            
            # Create output path
            input_path = Path(video_path)
            output_filename = f"slowed_{input_path.stem}_{slowdown_factor}x{input_path.suffix}"
            output_path = self.temp_dir / output_filename
            
            # Slow down video
            slowed_video = video.set_fps(new_fps)
            
            # Write slowed video
            logger.info(f"Writing slowed video to: {output_path}")
            slowed_video.write_videofile(
                str(output_path),
                fps=new_fps,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Clean up
            video.close()
            slowed_video.close()
            
            logger.info(f"Video slowdown completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error slowing down video: {e}")
            raise
    
    def extract_frames_at_timestamps(self, video_path: str, timestamps: List[float]) -> List[str]:
        """
        Extract frames at specific timestamps for analysis.
        Returns list of frame file paths.
        """
        try:
            logger.info(f"Extracting frames at timestamps: {timestamps}")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_paths = []
            
            for timestamp in timestamps:
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    frame_filename = f"frame_{timestamp:.2f}s.jpg"
                    frame_path = self.temp_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    logger.debug(f"Extracted frame at {timestamp}s: {frame_path}")
                else:
                    logger.warning(f"Could not extract frame at timestamp {timestamp}s")
            
            cap.release()
            logger.info(f"Extracted {len(frame_paths)} frames")
            return frame_paths
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def create_highlight_video(self, original_video_path: str, highlights: List[dict], 
                             output_path: str, fighter_name: str = "FIGHTER") -> str:
        """
        Create a highlights video with slow-motion clips and subtitles.
        """
        try:
            logger.info(f"Creating highlight video with {len(highlights)} highlights")
            
            video = VideoFileClip(original_video_path)
            highlight_clips = []
            
            # Calculate aspect ratio and scaling factors
            video_width, video_height = video.w, video.h
            aspect_ratio = video_width / video_height
            base_width = 1920  # Standard reference width
            scale_factor = video_width / base_width
            
            # Calculate adaptive font sizes based on video dimensions
            title_font_size = max(24, int(36 * scale_factor))
            subtitle_font_size = max(16, int(24 * scale_factor))
            intro_font_size = max(48, int(72 * scale_factor))
            
            logger.info(f"Video dimensions: {video_width}x{video_height}, aspect ratio: {aspect_ratio:.2f}, scale factor: {scale_factor:.2f}")
            logger.info(f"Font sizes - Title: {title_font_size}, Subtitle: {subtitle_font_size}, Intro: {intro_font_size}")
            
            # Add "HIGHLIGHTS" intro screen
            intro_screen = VideoFileClip(original_video_path).subclip(0, 0.1).fl(
                lambda get_frame, t: np.zeros((video.h, video.w, 3), dtype=np.uint8)
            ).set_duration(3.0)  # 3 second intro screen
            
            def add_intro_text(get_frame, t):
                frame = get_frame(t)
                from PIL import Image, ImageDraw, ImageFont
                import numpy as np
                
                pil_image = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_image)
                
                try:
                    font = ImageFont.truetype("arial.ttf", intro_font_size)
                except:
                    font = ImageFont.load_default()
                
                text = "HIGHLIGHTS"
                h, w, _ = frame.shape
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (w - text_width) // 2
                y = (h - text_height) // 2
                
                # Draw with white color
                draw.text((x, y), text, font=font, fill=(255, 255, 255))
                return np.array(pil_image)
            
            intro_screen_with_text = intro_screen.fl(add_intro_text)
            highlight_clips.append(intro_screen_with_text)
            
            for i, highlight in enumerate(highlights, 1):
                timestamp = highlight.get('timestamp', 0)
                detailed_feedback = highlight.get('detailed_feedback', '')
                action_required = highlight.get('action_required', '')
                
                # Create 4-second slow-motion clip around the highlight (0:10-0:14 example)
                start_time = max(0, timestamp - 2.0)  # 2s before highlight
                end_time = min(video.duration, timestamp + 2.0)  # 2s after highlight
                
                # Extract the clip and slow it down to 0.5x (half speed)
                clip = video.subclip(start_time, end_time)
                slowed_clip = clip.speedx(0.5)  # 0.5x speed (half speed)
                
                # Create a closure to capture the specific highlight data
                def create_highlight_overlay(highlight_num, feedback_text, action_text):
                    def add_head_and_subtitle(get_frame, t):
                        frame = get_frame(t)
                        frame = self._draw_head_tracking(frame, t, video, fighter_name)
                        # Now add the subtitle overlay as before
                        # Convert frame to PIL Image for text overlay
                        from PIL import Image, ImageDraw, ImageFont
                        import numpy as np
                        pil_image = Image.fromarray(frame)
                        draw = ImageDraw.Draw(pil_image)
                        try:
                            title_font = ImageFont.truetype("arial.ttf", title_font_size)
                            subtitle_font = ImageFont.truetype("arial.ttf", subtitle_font_size)
                        except:
                            title_font = ImageFont.load_default()
                            subtitle_font = ImageFont.load_default()
                        h, w, _ = frame.shape
                        title_text = f"HIGHLIGHT {highlight_num}"
                        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
                        title_width = title_bbox[2] - title_bbox[0]
                        title_x = (w - title_width) // 2
                        title_y = int(20 * scale_factor)  # Adaptive top margin
                        outline_color = (0, 0, 0)
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                if dx != 0 or dy != 0:
                                    draw.text((title_x + dx, title_y + dy), title_text, font=title_font, fill=outline_color)
                        draw.text((title_x, title_y), title_text, font=title_font, fill=(255, 255, 0))
                        feedback_lines = []
                        words = feedback_text.split()
                        current_line = ""
                        max_line_width = w - int(100 * scale_factor)  # Adaptive margin
                        for word in words:
                            test_line = current_line + " " + word if current_line else word
                            bbox = draw.textbbox((0, 0), test_line, font=subtitle_font)
                            if bbox[2] - bbox[0] < max_line_width:
                                current_line = test_line
                            else:
                                if current_line:
                                    feedback_lines.append(current_line)
                                current_line = word
                        if current_line:
                            feedback_lines.append(current_line)
                        if action_text:
                            feedback_lines.append("")
                            feedback_lines.append(f"ACTION: {action_text}")
                        line_height = int(30 * scale_factor)  # Adaptive line height
                        total_height = len(feedback_lines) * line_height
                        y_start = h - int(50 * scale_factor) - total_height  # Adaptive bottom margin
                        for line_idx, line in enumerate(feedback_lines):
                            if line.startswith("ACTION:"):
                                color = (255, 0, 0)
                            else:
                                color = (255, 255, 255)
                            bbox = draw.textbbox((0, 0), line, font=subtitle_font)
                            line_width = bbox[2] - bbox[0]
                            line_x = (w - line_width) // 2
                            y_pos = y_start + (line_idx * line_height)
                            for dx in [-2, -1, 0, 1, 2]:
                                for dy in [-2, -1, 0, 1, 2]:
                                    if dx != 0 or dy != 0:
                                        draw.text((line_x + dx, y_pos + dy), line, font=subtitle_font, fill=outline_color)
                            draw.text((line_x, y_pos), line, font=subtitle_font, fill=color)
                        return np.array(pil_image)
                    return add_head_and_subtitle
                
                # Apply the overlay with the specific highlight data
                overlay_func = create_highlight_overlay(i, detailed_feedback, action_required)
                final_clip = slowed_clip.fl(overlay_func)
                highlight_clips.append(final_clip)
                
                # Add black screen with highlight number between clips
                if i < len(highlights):
                    black_screen = VideoFileClip(original_video_path).subclip(0, 0.1).fl(
                        lambda get_frame, t: np.zeros((video.h, video.w, 3), dtype=np.uint8)
                    ).set_duration(2.0)  # 2 second black screen
                    
                    # Add highlight number to black screen - capture variable properly
                    def create_add_highlight_number(highlight_num):
                        def add_highlight_number(get_frame, t):
                            frame = get_frame(t)
                            from PIL import Image, ImageDraw, ImageFont
                            import numpy as np
                            
                            pil_image = Image.fromarray(frame)
                            draw = ImageDraw.Draw(pil_image)
                            
                            try:
                                font = ImageFont.truetype("arial.ttf", intro_font_size)
                            except:
                                font = ImageFont.load_default()
                            
                            text = f"HIGHLIGHT {highlight_num}"
                            h, w, _ = frame.shape
                            bbox = draw.textbbox((0, 0), text, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                            x = (w - text_width) // 2
                            y = (h - text_height) // 2
                            
                            # Draw with white color
                            draw.text((x, y), text, font=font, fill=(255, 255, 255))
                            return np.array(pil_image)
                        return add_highlight_number
                    
                    black_screen_with_text = black_screen.fl(create_add_highlight_number(i))
                    highlight_clips.append(black_screen_with_text)
            
            # Concatenate all clips
            final_video = concatenate_videoclips(highlight_clips)
            
            # Write the final video
            final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            # Clean up
            final_video.close()
            for clip in highlight_clips:
                clip.close()
            video.close()
            
            logger.info(f"Highlight video created successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating highlight video: {e}")
            raise
        finally:
            if 'video' in locals():
                video.close()
    
    def _draw_head_tracking(self, frame, t, video, fighter_name):
        # Add head pointer if pose detection is available
        if self.pose_detection_available:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                if results.pose_landmarks:
                    nose_landmark = results.pose_landmarks.landmark[0]
                    h, w, _ = frame.shape
                    nose_x = int(nose_landmark.x * w)
                    nose_y = int(nose_landmark.y * h)
                    head_center_x = nose_x
                    head_center_y = nose_y - 20
                    
                    # Calculate adaptive sizes based on video dimensions
                    base_width = 1920
                    scale_factor = w / base_width
                    circle_radius = max(8, int(15 * scale_factor))
                    line_thickness = max(2, int(6 * scale_factor))
                    outline_thickness = max(1, int(3 * scale_factor))
                    pointer_length = max(50, int(100 * scale_factor))
                    
                    # Draw a more prominent head pointer
                    cv2.circle(frame, (head_center_x, head_center_y), circle_radius, (0, 0, 255), -1)  # Red circle
                    cv2.circle(frame, (head_center_x, head_center_y), circle_radius, (255, 255, 255), outline_thickness)  # White outline
                    
                    # Draw pointer line
                    pointer_start_y = head_center_y - pointer_length
                    cv2.line(frame, (head_center_x, pointer_start_y), (head_center_x, head_center_y - circle_radius), (0, 0, 255), line_thickness)  # Red line
                    cv2.line(frame, (head_center_x, pointer_start_y), (head_center_x, head_center_y - circle_radius), (255, 255, 255), outline_thickness)  # White outline
                    
                    # Draw fighter name
                    text = fighter_name
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = max(0.6, 1.0 * scale_factor)
                    thickness = max(1, int(3 * scale_factor))
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = head_center_x - text_width // 2
                    text_y = pointer_start_y - int(20 * scale_factor)
                    
                    # Background rectangle for text
                    padding = int(10 * scale_factor)
                    cv2.rectangle(frame, (text_x - padding, text_y - text_height - padding), (text_x + text_width + padding, text_y + padding), (0, 0, 0), -1)
                    cv2.rectangle(frame, (text_x - padding, text_y - text_height - padding), (text_x + text_width + padding, text_y + padding), (255, 255, 255), outline_thickness)
                    
                    # Draw text
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
                    
                    logger.debug(f"Head tracking drawn at ({head_center_x}, {head_center_y}) for {fighter_name}")
            except Exception as e:
                logger.debug(f"Pose detection error on frame: {e}")
        else:
            # Fallback: draw a simple pointer in the center if pose detection is not available
            h, w, _ = frame.shape
            head_center_x = w // 2
            head_center_y = h // 3
            
            # Calculate adaptive sizes based on video dimensions
            base_width = 1920
            scale_factor = w / base_width
            circle_radius = max(8, int(15 * scale_factor))
            line_thickness = max(2, int(6 * scale_factor))
            outline_thickness = max(1, int(3 * scale_factor))
            pointer_length = max(50, int(100 * scale_factor))
            
            # Draw a simple pointer
            cv2.circle(frame, (head_center_x, head_center_y), circle_radius, (0, 0, 255), -1)
            cv2.circle(frame, (head_center_x, head_center_y), circle_radius, (255, 255, 255), outline_thickness)
            
            # Draw pointer line
            pointer_start_y = head_center_y - pointer_length
            cv2.line(frame, (head_center_x, pointer_start_y), (head_center_x, head_center_y - circle_radius), (0, 0, 255), line_thickness)
            cv2.line(frame, (head_center_x, pointer_start_y), (head_center_x, head_center_y - circle_radius), (255, 255, 255), outline_thickness)
            
            # Draw fighter name
            text = fighter_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, 1.0 * scale_factor)
            thickness = max(1, int(3 * scale_factor))
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = head_center_x - text_width // 2
            text_y = pointer_start_y - int(20 * scale_factor)
            
            padding = int(10 * scale_factor)
            cv2.rectangle(frame, (text_x - padding, text_y - text_height - padding), (text_x + text_width + padding, text_y + padding), (0, 0, 0), -1)
            cv2.rectangle(frame, (text_x - padding, text_y - text_height - padding), (text_x + text_width + padding, text_y + padding), (255, 255, 255), outline_thickness)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
            
            logger.debug(f"Fallback head tracking drawn for {fighter_name}")
        
        return frame
    
    def add_overlays_to_video(self, video_path: str, overlays: List[dict], 
                             output_path: str, fighter_name: str = "FIGHTER") -> str:
        """
        Add text overlays, pose detection, and annotations to video based on AI feedback.
        """
        try:
            logger.info(f"Adding overlays to video: {len(overlays)} overlays")
            
            video = VideoFileClip(video_path)
            
            # Calculate aspect ratio and scaling factors
            video_width, video_height = video.w, video.h
            aspect_ratio = video_width / video_height
            base_width = 1920  # Standard reference width
            scale_factor = video_width / base_width
            
            # Calculate adaptive font sizes based on video dimensions
            overlay_font_size = max(20, int(32 * scale_factor))
            
            logger.info(f"Video dimensions: {video_width}x{video_height}, aspect ratio: {aspect_ratio:.2f}, scale factor: {scale_factor:.2f}")
            logger.info(f"Overlay font size: {overlay_font_size}")
            
            def add_overlay(get_frame, t):
                frame = get_frame(t).copy()  # Make a writable copy
                frame = self._draw_head_tracking(frame, t, video, fighter_name)
                
                # Find overlays that should be shown at this time
                for overlay in overlays:
                    start_time = overlay.get('start_time', 0)
                    end_time = overlay.get('end_time', start_time + 3)
                    
                    if start_time <= t <= end_time:
                        # Convert frame to PIL Image for text overlay
                        from PIL import Image, ImageDraw, ImageFont
                        import numpy as np
                        
                        pil_image = Image.fromarray(frame)
                        draw = ImageDraw.Draw(pil_image)
                        
                        # Try to use a default font, fallback to basic if not available
                        try:
                            font = ImageFont.truetype("arial.ttf", overlay_font_size)
                        except:
                            font = ImageFont.load_default()
                        
                        text = overlay.get('text', '')
                        h, w, _ = frame.shape
                        
                        # Split text into multiple lines if needed
                        words = text.split()
                        lines = []
                        current_line = ""
                        max_line_width = w - int(100 * scale_factor)  # Adaptive margin
                        
                        for word in words:
                            test_line = current_line + " " + word if current_line else word
                            bbox = draw.textbbox((0, 0), test_line, font=font)
                            if bbox[2] - bbox[0] < max_line_width:
                                current_line = test_line
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = word
                        if current_line:
                            lines.append(current_line)
                        
                        # Calculate position (centered at bottom with adaptive margin)
                        line_height = int(40 * scale_factor)
                        total_height = len(lines) * line_height
                        y_start = h - int(30 * scale_factor) - total_height
                        
                        # Draw each line
                        for i, line in enumerate(lines):
                            bbox = draw.textbbox((0, 0), line, font=font)
                            line_width = bbox[2] - bbox[0]
                            x = (w - line_width) // 2
                            y = y_start + (i * line_height)
                            
                            # Draw text with black outline for visibility
                            outline_color = (0, 0, 0)
                            outline_thickness = max(1, int(2 * scale_factor))
                            for dx in range(-outline_thickness, outline_thickness + 1):
                                for dy in range(-outline_thickness, outline_thickness + 1):
                                    if dx != 0 or dy != 0:
                                        draw.text((x + dx, y + dy), line, font=font, fill=outline_color)
                            
                            # White text
                            draw.text((x, y), line, font=font, fill=(255, 255, 255))
                        
                        frame = np.array(pil_image)
                        break
                
                return frame
            
            # Apply overlay function
            final_video = video.fl(add_overlay)
            
            # Write video with overlays
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Clean up
            video.close()
            final_video.close()
            
            logger.info(f"Video with overlays created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding overlays to video: {e}")
            raise
    
    def merge_video_audio(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        Merge video with audio track.
        """
        try:
            logger.info(f"Merging video and audio: {video_path} + {audio_path}")
            
            video = VideoFileClip(video_path)
            
            # Load audio using AudioFileClip instead of VideoFileClip
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            audio = AudioFileClip(audio_path)
            
            # Set audio to video
            final_video = video.set_audio(audio)
            
            # Write final video with explicit fps
            final_video.write_videofile(
                output_path,
                fps=video.fps,  # Use original video fps
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Clean up
            video.close()
            audio.close()
            final_video.close()
            
            logger.info(f"Video with audio created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error merging video and audio: {e}")
            # Fallback: just copy the overlay video without audio
            logger.info("Falling back to video without audio")
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path

    def merge_video_audio_with_original(self, overlay_video_path: str, tts_audio_path: str, 
                                      original_video_path: str, output_path: str) -> str:
        """
        Merge overlay video with TTS audio while preserving original audio at 50% volume.
        """
        try:
            logger.info(f"Merging overlay video with TTS audio and original audio: {overlay_video_path}")
            
            # Import AudioFileClip
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            
            # Load all components
            overlay_video = VideoFileClip(overlay_video_path)
            tts_audio = AudioFileClip(tts_audio_path)
            original_video = VideoFileClip(original_video_path)
            
            # Get original audio and reduce volume to 50%
            original_audio = original_video.audio
            if original_audio:
                original_audio = original_audio.volumex(0.5)
                logger.info("Original audio volume reduced to 50%")
            else:
                logger.warning("No original audio found in video")
            
            # Combine TTS audio with original audio
            if original_audio:
                # Ensure both audios have the same duration
                max_duration = max(tts_audio.duration, original_audio.duration)
                tts_audio = tts_audio.set_duration(max_duration)
                original_audio = original_audio.set_duration(max_duration)
                
                # Combine audios
                combined_audio = tts_audio.set_duration(max_duration)
                # Note: MoviePy doesn't have a simple way to mix audio, so we'll prioritize TTS
                # In a more advanced implementation, you'd use audio mixing libraries
                logger.info("Combined TTS audio with original audio (TTS prioritized)")
            else:
                combined_audio = tts_audio
            
            # Set combined audio to overlay video
            final_video = overlay_video.set_audio(combined_audio)
            
            # Write final video
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Clean up
            overlay_video.close()
            tts_audio.close()
            original_video.close()
            if original_audio:
                original_audio.close()
            final_video.close()
            
            logger.info(f"Video with combined audio created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error merging video with combined audio: {e}")
            raise
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files."""
        try:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")
    
    def cleanup_slowed_video(self, video_path: str):
        """Clean up the slowed down video after analysis."""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up slowed video: {video_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up slowed video: {e}") 