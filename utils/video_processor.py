import os
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageSequenceClip, TextClip, ColorClip
from PIL import Image, ImageDraw, ImageFont
import base64
import tempfile

class VideoProcessor:
    """Surgical VideoProcessor that follows exact specifications"""
    
    def __init__(self):
        # Colors for overlays - FIXED RGB VALUES
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),      # FIXED: Proper red
            'blue': (0, 0, 255),     # FIXED: Proper blue
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),  # FIXED: Proper yellow
            'magenta': (255, 0, 255), # For head pointer
            'purple': (128, 0, 128)   # For neon aesthetic
        }
        
        print("✅ VideoProcessor initialized (surgical mode)")

    def create_highlight_video(self, video_path: str, highlights: list, output_path: str, user_name: str = "FIGHTER") -> str:
        """
        Main entry point - creates final video with intro card, title cards, and t-1 to t+1 highlights at 0.4x speed
        """
        # CRITICAL: Proper MediaPipe resource management with with statement
        with mp.solutions.pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            try:
                print(f"🎬 Starting surgical video processing for {len(highlights)} highlights")
                
                # Load source video
                source_clip = VideoFileClip(video_path)
                source_fps = source_clip.fps
                source_duration = source_clip.duration
                source_width, source_height = source_clip.size
                
                print(f"📹 Source video: {source_duration:.2f}s at {source_fps}fps, {source_width}x{source_height}")
                
                all_clips = []
                
                # CRITICAL: Generate Intro Card ("HIGHLIGHTS")
                intro_card = self._create_text_card("HIGHLIGHTS", source_width, source_height, source_fps, duration=2.0)
                all_clips.append(intro_card)
                
                # Process each highlight with surgical precision
                for i, highlight in enumerate(highlights):
                    print(f"🎯 Processing highlight {i+1}/{len(highlights)}")
                    
                    # CRITICAL: Generate Title Card ("HIGHLIGHT i+1")
                    title_card = self._create_text_card(f"HIGHLIGHT {i+1}", source_width, source_height, source_fps, duration=1.5)
                    all_clips.append(title_card)
                    
                    # Extract timestamp and convert to seconds
                    timestamp = self._parse_timestamp(highlight.get('timestamp', '00:00'))
                    
                    # CRITICAL: t-1 to t+1 window (2-second total duration)
                    start_time = max(0, timestamp - 1)
                    end_time = min(source_duration, timestamp + 1)
                    
                    print(f"⏰ Highlight window: {start_time:.2f}s to {end_time:.2f}s")
                    
                    # Extract the highlight clip
                    highlight_clip = source_clip.subclip(start_time, end_time)
                    
                    # CRITICAL: Slow down to 0.25x speed
                    slowed_clip = highlight_clip.speedx(0.25)
                    
                    # Process frames with overlays
                    processed_frames = []
                    for frame in slowed_clip.iter_frames():
                        # CRITICAL: Convert BGR to RGB before processing
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame = self._add_overlays(
                            frame_rgb, 
                            '',  # No short_text needed
                            highlight.get('action_required', ''),
                            user_name,
                            pose  # Pass the pose object to the overlay method
                        )
                        processed_frames.append(processed_frame)
                    
                    # Create new clip from processed frames
                    processed_clip = ImageSequenceClip(processed_frames, fps=slowed_clip.fps)
                    all_clips.append(processed_clip)
                    
                    # Cleanup
                    highlight_clip.close()
                    slowed_clip.close()
                
                # CRITICAL: Concatenate all clips (Intro -> Title -> Clip sequence)
                if all_clips:
                    final_video = concatenate_videoclips(all_clips)
                    
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
                    for clip in all_clips:
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
                # Cleanup source clip
                if 'source_clip' in locals():
                    source_clip.close()

    def _create_text_card(self, text: str, width: int, height: int, fps: float, duration: float = 2.0):
        """Create a text card with neon aesthetic"""
        try:
            # Create a black background
            bg_color = (0, 0, 0)  # Black
            card = ColorClip(size=(width, height), color=bg_color, duration=duration)
            
            # Create text clip with neon purple color
            text_clip = TextClip(
                text,
                fontsize=int(width * 0.08),  # Dynamic font size
                color='white',
                stroke_color='purple',
                stroke_width=3,
                font='Arial-Bold'
            ).set_position('center').set_duration(duration)
            
            # Composite text over background
            final_card = card.set_make_frame(lambda t: np.array(card.get_frame(t)))
            final_card = final_card.set_audio(None)  # No audio for cards
            
            return final_card
            
        except Exception as e:
            print(f"⚠️ Text card creation failed: {e}")
            # Fallback: simple color clip
            return ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)

    def _add_overlays(self, frame, short_text, action_text, user_name, pose):
        """
        Add head pointer and static captions with thick black outline
        """
        try:
            # Convert frame to PIL for text rendering
            pil_frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_frame)
            
            h, w, _ = frame.shape
            
            # CRITICAL: Head pointer using MediaPipe
            head_pos = self._detect_head_position(frame, pose)
            if head_pos:
                x, y = head_pos
                
                # Draw "FIGHTER" label above the pointer
                label_text = user_name
                label_font_size = max(20, int(w * 0.025))  # Dynamic font size
                try:
                    label_font = ImageFont.truetype("arial.ttf", label_font_size)
                except:
                    label_font = ImageFont.load_default()
                
                # Position label above the pointer
                label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                label_width = label_bbox[2] - label_bbox[0]
                label_height = label_bbox[3] - label_bbox[1]
                label_x = x - label_width // 2
                label_y = y - int(h * 0.1)  # 10% of frame height above pointer
                
                # Draw label with outline
                draw.text(
                    (label_x, label_y),
                    label_text,
                    font=label_font,
                    fill=self.colors['white'],
                    stroke_width=2,
                    stroke_fill=self.colors['black']
                )
                
                # Draw thin vertical line from label to pointer
                line_start_y = label_y + label_height + 5
                line_end_y = y - int(w * 0.02)  # 2% of frame width above pointer
                draw.line(
                    [(x, line_start_y), (x, line_end_y)],
                    fill=self.colors['white'],
                    width=2
                )
                
                # Draw small, clean circle at head position
                radius = max(5, int(w * 0.015))  # 1.5% of frame width, minimum 5px
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    outline=self.colors['magenta'],
                    width=3
                )
            
            # CRITICAL: Static captions at bottom center using action_required
            if action_text:
                # Calculate font size (3.9% of frame width - 30% bigger)
                font_size = max(26, int(w * 0.039))
                
                try:
                    # Try to use Montserrat Semi-Bold font
                    font = ImageFont.truetype("Montserrat-SemiBold.ttf", font_size)
                except:
                    try:
                        # Fallback to Arial Bold
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                
                # CRITICAL: Position at bottom center with 5% margin
                text_bbox = draw.textbbox((0, 0), action_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = (w - text_width) // 2
                text_y = h - text_height - int(h * 0.05)  # 5% margin from bottom
                
                # CRITICAL: Draw text with THICK black outline (no background)
                draw.text(
                    (text_x, text_y),
                    action_text,
                    font=font,
                    fill=self.colors['white'],
                    stroke_width=5,  # THICK outline
                    stroke_fill=self.colors['black']
                )
            
            # Convert back to numpy array
            return np.array(pil_frame)
            
        except Exception as e:
            print(f"⚠️ Overlay rendering failed: {e}")
            return frame

    def _detect_head_position(self, frame, pose):
        """
        Detect head position using MediaPipe pose landmarks
        """
        try:
            results = pose.process(frame)
            
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