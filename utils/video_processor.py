import os
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageSequenceClip, TextClip, ColorClip, CompositeVideoClip, CompositeAudioClip
from PIL import Image, ImageDraw, ImageFont
import base64
import tempfile
import textwrap

class VideoProcessor:
    """Surgical VideoProcessor with correct audio sync pipeline"""
    
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
        
        # Initialize TTS client for individual highlight audio
        self.tts_client = self._init_tts_client()
        
        print("✅ VideoProcessor initialized (surgical mode with audio sync)")

    def _init_tts_client(self):
        """Initialize TTS client for individual highlight audio generation"""
        try:
            import os
            from elevenlabs import generate, save, set_api_key
            
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                set_api_key(api_key)
                return {"generate": generate, "save": save}
            else:
                print("⚠️ No ElevenLabs API key found, TTS will be disabled")
                return None
        except ImportError:
            print("⚠️ ElevenLabs not available, TTS will be disabled")
            return None

    def create_highlight_video(self, video_path: str, highlights: list, output_path: str, user_name: str = "FIGHTER") -> str:
        """
        Main entry point - creates final video with intro card, title cards, and individual highlight clips with audio
        """
        try:
            print(f"🎬 Starting surgical video processing for {len(highlights)} highlights")
            
            # Load source video
            source_clip = VideoFileClip(video_path)
            source_fps = source_clip.fps
            source_duration = source_clip.duration
            source_width, source_height = source_clip.size
            
            print(f"📹 Source video: {source_duration:.2f}s at {source_fps}fps, {source_width}x{source_height}")
            
            # CRITICAL: Create empty list for final clips
            final_clips = []
            
            # CRITICAL: Generate 2-second Intro Title Card ("HIGHLIGHTS")
            print("🎬 Creating intro card...")
            intro_card = self._create_text_card("HIGHLIGHTS", source_width, source_height, source_fps, duration=2.0)
            final_clips.append(intro_card)
            print("✅ Intro card added to final_clips")
            
            # CRITICAL: Process each highlight with individual TTS
            for i, highlight in enumerate(highlights):
                print(f"🎯 Processing highlight {i+1}/{len(highlights)}")
                
                # CRITICAL: Generate 1.5-second Highlight Title Card
                print(f"🎬 Creating title card for highlight {i+1}...")
                title_card = self._create_text_card(f"HIGHLIGHT {i+1}", source_width, source_height, source_fps, duration=1.5)
                final_clips.append(title_card)
                print(f"✅ Title card {i+1} added to final_clips")
                
                # CRITICAL: Generate individual highlight clip with its own audio
                highlight_clip = self._create_single_highlight_clip(
                    source_clip, 
                    highlight, 
                    user_name, 
                    source_fps
                )
                final_clips.append(highlight_clip)
                print(f"✅ Highlight clip {i+1} added to final_clips")
            
            # CRITICAL: Concatenate all clips in correct order
            if final_clips:
                print(f"🎬 Concatenating {len(final_clips)} clips...")
                final_video = concatenate_videoclips(final_clips)
                
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
                for clip in final_clips:
                    clip.close()
                
                print(f"✅ Surgical video processing complete: {output_path}")
                return output_path
            else:
                raise ValueError("No valid clips were processed")
                
        except Exception as e:
            print(f"❌ Surgical video processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: copy original video
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path
        finally:
            # Cleanup source clip
            if 'source_clip' in locals():
                source_clip.close()

    def _create_single_highlight_clip(self, source_clip, highlight, user_name, source_fps):
        """
        CRITICAL: Create a single highlight clip with its own TTS audio attached
        This is the core fix for audio sync issues
        """
        try:
            # Step 1: Generate TTS for this specific highlight
            audio_path = None
            if self.tts_client:
                audio_path = self._generate_highlight_tts(highlight, f"highlight_{hash(str(highlight))}")
            
            # Step 2: Generate video with overlays (silent)
            video_clip = self._generate_highlight_video(source_clip, highlight, user_name, source_fps)
            
            # Step 3: Attach audio to video if available with 1-second delay
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_clip = AudioFileClip(audio_path)
                    
                    # CRITICAL: Create 1-second silence and prepend to TTS audio
                    from moviepy.audio.AudioClip import AudioClip
                    silence = AudioClip(make_frame=lambda t: [0, 0], duration=1, fps=44100)
                    
                    # Prepend the silence to the TTS audio
                    final_audio = CompositeAudioClip([silence, audio_clip.set_start(1)])
                    
                    # Set the new composite audio to the video
                    final_clip = video_clip.set_audio(final_audio)
                    print(f"✅ Audio attached to highlight clip with 1-second delay")
                    return final_clip
                except Exception as e:
                    print(f"⚠️ Audio attachment failed: {e}, returning silent clip")
                    return video_clip
            else:
                print(f"⚠️ No audio available for highlight, returning silent clip")
                return video_clip
                
        except Exception as e:
            print(f"❌ Highlight clip creation failed: {e}")
            # Return a simple fallback clip
            return self._create_fallback_clip(source_clip, highlight, source_fps)

    def _generate_highlight_tts(self, highlight, clip_id):
        """Generate TTS audio for a single highlight with sentence-by-sentence pacing"""
        try:
            if not self.tts_client:
                return None
            
            # Use only action_required text for TTS
            action_text = highlight.get('action_required', '')
            if not action_text:
                return None
            
            print(f"🔊 Generating paced TTS for: {action_text[:50]}...")
            
            # Create audio directory if it doesn't exist
            os.makedirs("output/audio", exist_ok=True)
            audio_path = f"output/audio/{clip_id}.mp3"
            
            # Split text into sentences for paced narration
            sentences = [s.strip() for s in action_text.split('.') if s.strip()]
            
            if not sentences:
                return None
            
            # Create empty list for audio clips
            audio_clips = []
            
            # Generate TTS for each sentence individually
            for i, sentence in enumerate(sentences):
                print(f"🔊 Generating TTS for sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
                
                try:
                    # Generate TTS for this sentence
                    audio = self.tts_client["generate"](
                        text=sentence,
                        voice="21m00Tcm4TlvDq8ikWAM",
                        model="eleven_monolingual_v1"
                    )
                    
                    # Save individual sentence audio temporarily
                    temp_audio_path = f"output/audio/{clip_id}_sentence_{i}.mp3"
                    self.tts_client["save"](audio, temp_audio_path)
                    
                    # Load as AudioFileClip
                    sentence_clip = AudioFileClip(temp_audio_path)
                    audio_clips.append(sentence_clip)
                    
                    # Add silence after each sentence (except the last one)
                    if i < len(sentences) - 1:
                        # Create 0.4 seconds of silence
                        silence_duration = 0.4
                        silence_clip = AudioFileClip(temp_audio_path).set_duration(silence_duration)
                        silence_clip = silence_clip.volumex(0)  # Mute the audio to create silence
                        audio_clips.append(silence_clip)
                    
                    # Clean up temp file
                    os.remove(temp_audio_path)
                    
                except Exception as e:
                    print(f"⚠️ TTS generation failed for sentence {i+1}: {e}")
                    continue
            
            if not audio_clips:
                print("⚠️ No audio clips generated")
                return None
            
            # Concatenate all audio clips (speech + silence + speech + silence...)
            print(f"🔊 Concatenating {len(audio_clips)} audio clips...")
            final_audio = CompositeAudioClip(audio_clips)
            
            # Save final composite audio
            final_audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Clean up audio clips
            for clip in audio_clips:
                clip.close()
            final_audio.close()
            
            print(f"✅ Paced TTS generated: {audio_path}")
            return audio_path
            
        except Exception as e:
            print(f"⚠️ TTS generation failed: {e}")
            return None

    def _generate_highlight_video(self, source_clip, highlight, user_name, source_fps):
        """Generate video clip with overlays for a single highlight"""
        try:
            # Extract timestamp and convert to seconds
            timestamp = self._parse_timestamp(highlight.get('timestamp', '00:00'))
            
            # CRITICAL: t-1 to t+1 window (2-second total duration)
            start_time = max(0, timestamp - 1)
            end_time = min(source_clip.duration, timestamp + 1)
            
            print(f"⏰ Highlight window: {start_time:.2f}s to {end_time:.2f}s")
            
            # Extract the highlight clip
            highlight_clip = source_clip.subclip(start_time, end_time)
            
            # CRITICAL: Slow down to 0.25x speed
            slowed_clip = highlight_clip.speedx(0.25)
            
            # CRITICAL: Process frames with overlays using MediaPipe
            processed_frames = []
            with mp.solutions.pose.Pose(
                static_image_mode=False, 
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:
                
                for frame in slowed_clip.iter_frames():
                    # CRITICAL: MoviePy iter_frames() already returns RGB, NO conversion needed
                    # DELETED: frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame = self._add_overlays(
                        frame,  # Pass original RGB frame directly
                        highlight.get('action_required', ''),
                        user_name,
                        pose
                    )
                    processed_frames.append(processed_frame)
            
            # Create new clip from processed frames
            processed_clip = ImageSequenceClip(processed_frames, fps=slowed_clip.fps)
            
            # Cleanup
            highlight_clip.close()
            slowed_clip.close()
            
            return processed_clip
            
        except Exception as e:
            print(f"❌ Highlight video generation failed: {e}")
            return self._create_fallback_clip(source_clip, highlight, source_fps)

    def _create_fallback_clip(self, source_clip, highlight, source_fps):
        """Create a simple fallback clip if processing fails"""
        try:
            # Create a simple 2-second black clip with text
            width, height = source_clip.size
            fallback_clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=2.0)
            
            # Add text overlay
            action_text = highlight.get('action_required', 'Analysis failed')
            text_clip = TextClip(
                action_text,
                fontsize=int(width * 0.03),
                color='white',
                font='Arial-Bold'
            ).set_position('center').set_duration(2.0)
            
            final_clip = fallback_clip.set_make_frame(lambda t: np.array(fallback_clip.get_frame(t)))
            return final_clip
            
        except Exception as e:
            print(f"❌ Fallback clip creation failed: {e}")
            # Return a simple black clip
            return ColorClip(size=(640, 480), color=(0, 0, 0), duration=2.0)

    def _create_text_card(self, text: str, width: int, height: int, fps: float, duration: float = 2.0):
        """Create a text card with neon aesthetic - FIXED to return proper CompositeVideoClip"""
        try:
            print(f"🎬 Creating text card: '{text}' for {duration}s at {width}x{height}")
            
            # Create a black background
            bg_color = (0, 0, 0)  # Black
            card = ColorClip(size=(width, height), color=bg_color, duration=duration)
            print(f"✅ Background card created: {card.size}")
            
            # Calculate dynamic font size
            font_size = int(width * 0.08)  # 8% of frame width
            print(f"📝 Using font size: {font_size}")
            
            # Create text clip with neon purple color
            try:
                text_clip = TextClip(
                    text,
                    fontsize=font_size,
                    color='white',
                    stroke_color='purple',
                    stroke_width=3,
                    font='Arial-Bold'
                ).set_position('center').set_duration(duration)
                print(f"✅ Text clip created: {text_clip.size}")
            except Exception as text_error:
                print(f"⚠️ Text clip creation failed: {text_error}")
                # Fallback to default font
                text_clip = TextClip(
                    text,
                    fontsize=font_size,
                    color='white',
                    stroke_color='purple',
                    stroke_width=3
                ).set_position('center').set_duration(duration)
                print(f"✅ Text clip created with fallback font: {text_clip.size}")
            
            # CRITICAL: Return proper CompositeVideoClip
            final_card = CompositeVideoClip([card, text_clip])
            print(f"✅ Text card '{text}' created successfully - Final size: {final_card.size}")
            
            # Verify the card has content by checking a frame
            try:
                test_frame = final_card.get_frame(1.0)  # Get frame at 1 second
                if test_frame is not None:
                    print(f"✅ Test frame retrieved successfully - Shape: {test_frame.shape}")
                else:
                    print("⚠️ Test frame is None")
            except Exception as frame_error:
                print(f"⚠️ Test frame retrieval failed: {frame_error}")
            
            return final_card
            
        except Exception as e:
            print(f"❌ Text card creation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: simple color clip with text
            try:
                fallback_card = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
                fallback_text = TextClip(
                    text,
                    fontsize=30,
                    color='white'
                ).set_position('center').set_duration(duration)
                return CompositeVideoClip([fallback_card, fallback_text])
            except:
                return ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)

    def _add_overlays(self, frame, action_text, user_name, pose):
        """
        CRITICAL: Add head pointer and static captions with proper color conversion and dynamic sizing
        """
        try:
            # CRITICAL: Convert frame to PIL for text rendering (already RGB from MoviePy)
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
            
            # CRITICAL: Static captions with text wrapping, 30% from bottom, 30% bigger
            if action_text:
                # CRITICAL: Text wrapping to prevent spilling
                wrapped_lines = textwrap.wrap(action_text, width=30)
                
                # Calculate font size (7% of frame width - 30% bigger than before)
                font_size = max(26, int(w * 0.07))
                
                try:
                    # Try to use Montserrat Semi-Bold font
                    font = ImageFont.truetype("Montserrat-SemiBold.ttf", font_size)
                except:
                    try:
                        # Fallback to Arial Bold
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                
                # Calculate total text height for all lines
                total_text_height = 0
                line_heights = []
                for line in wrapped_lines:
                    text_bbox = draw.textbbox((0, 0), line, font=font)
                    line_height = text_bbox[3] - text_bbox[1]
                    line_heights.append(line_height)
                    total_text_height += line_height
                
                # CRITICAL: Position at 30% from bottom of screen
                text_y = h - total_text_height - int(h * 0.30)
                
                # Draw each line of wrapped text
                current_y = text_y
                for i, line in enumerate(wrapped_lines):
                    text_bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_x = (w - text_width) // 2
                    
                    # CRITICAL: Draw text with THICK black outline (no background)
                    draw.text(
                        (text_x, current_y),
                        line,
                        font=font,
                        fill=self.colors['white'],
                        stroke_width=6,  # THICKER outline for better readability
                        stroke_fill=self.colors['black']
                    )
                    
                    # Move to next line
                    current_y += line_heights[i] + 5  # 5px spacing between lines
            
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
        Combine video with TTS audio using MoviePy (legacy method for compatibility)
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