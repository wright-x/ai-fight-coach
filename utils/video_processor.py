import os
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, TextClip, ColorClip, CompositeVideoClip, ImageClip, ImageSequenceClip
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
                
                # Add 1.5 second gap between highlights (except after the last one)
                if i < len(highlights) - 1:
                    print(f"⏸️ Adding 1.5s gap after highlight {i+1}...")
                    gap_clip = self._create_gap_clip(source_width, source_height, source_fps, duration=1.5)
                    final_clips.append(gap_clip)
                    print(f"✅ Gap clip added after highlight {i+1}")
            
            # --- INSERT THIS CODE BLOCK BEFORE `concatenate_videoclips` ---
            print(f"🛡️ Starting Final Resolution Safety Check...")
            enforced_clips = []
            target_size = (source_width, source_height) # The one and only correct size

            for i, clip in enumerate(final_clips):
                if clip.size != [target_size[0], target_size[1]]:
                    print(f"⚠️ Clip {i} has mismatched size {clip.size}. ENFORCING target size {target_size}.")
                    resized_clip = clip.resize(newsize=target_size)
                    enforced_clips.append(resized_clip)
                    clip.close() # Free memory from the old clip
                else:
                    enforced_clips.append(clip)

            print("✅ Safety Check complete. All clips conform to the correct resolution.")

            # Now, use the new, clean list for the final concatenation
            final_video = concatenate_videoclips(enforced_clips)

            # Clean up the enforced clips list
            for clip in enforced_clips:
                if clip in final_clips: continue # Avoid double-closing
                clip.close()

            # --- END OF THE NEW CODE BLOCK ---

            # CRITICAL: Video concatenation is now handled in the safety check above
                
                # Write final video with proper codec
                final_video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                # Cleanup
                final_video.close()
                source_clip.close()
                
                print(f"✅ Final video created: {output_path}")
                return output_path
            else:
                print("❌ No clips to concatenate")
                return None
                
        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

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
                    
                    # Create 1-second silent lead-in so voice never starts abruptly
                    from moviepy.audio.AudioClip import AudioClip
                    lead_in = AudioClip(lambda t: 0, duration=1)
                    
                    # Join lead-in and audio sequentially
                    from moviepy.editor import concatenate_audioclips
                    combined_audio = concatenate_audioclips([lead_in, audio_clip])
                    
                    # Set the combined audio to the video and sync duration
                    final_clip = video_clip.set_audio(combined_audio)
                    final_clip = final_clip.set_duration(combined_audio.duration)
                    
                    print(f"✅ Audio attached with 1-second lead-in")
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
        """
        CLEAN TTS SYSTEM: Generate speech for each highlight
        - Take action_required text
        - Break into sentences
        - Generate TTS for each sentence
        - Add 0.4s pause after each sentence
        - Join sequentially (no overlap)
        - Save as single WAV at 44.1kHz
        """
        try:
            if not self.tts_client:
                return None
            
            # Use only action_required text for TTS
            action_text = highlight.get('action_required', '')
            if not action_text:
                return None
            
            print(f"🔊 Generating clean TTS for: {action_text[:50]}...")
            
            # Create audio directory if it doesn't exist
            os.makedirs("output/audio", exist_ok=True)
            audio_path = f"output/audio/{clip_id}.wav"  # Use WAV format
            
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
                    
                    # Add 0.4 seconds of silence after each sentence (except the last one)
                    if i < len(sentences) - 1:
                        from moviepy.audio.AudioClip import AudioClip
                        silence_clip = AudioClip(lambda t: 0, duration=0.4)
                        audio_clips.append(silence_clip)
                    
                    # Clean up temp file
                    os.remove(temp_audio_path)
                    
                except Exception as e:
                    print(f"⚠️ TTS generation failed for sentence {i+1}: {e}")
                    continue
            
            if not audio_clips:
                print("⚠️ No audio clips generated")
                return None
            
            # Join all clips sequentially (no overlap)
            print(f"🔊 Joining {len(audio_clips)} audio clips sequentially...")
            from moviepy.editor import concatenate_audioclips
            final_audio = concatenate_audioclips(audio_clips)
            
            # Save final audio as WAV at 44.1kHz
            final_audio.write_audiofile(audio_path, fps=44100, verbose=False, logger=None)
            
            # Clean up audio clips
            for clip in audio_clips:
                clip.close()
            final_audio.close()
            
            print(f"✅ Clean TTS generated: {audio_path}")
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
                fontsize=int(width * 30),
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
        """
        Loads a pre-made image for the given title card, resizes it to match the
        source video's dimensions, and returns it as a video clip.
        """
        try:
            # 1. Determine which image to load based on the text.
            image_map = {
                "HIGHLIGHTS": "static/images/HIGHLIGHTS.png",
                "HIGHLIGHT 1": "static/images/HIGHLIGHTS (1).png",
                "HIGHLIGHT 2": "static/images/HIGHLIGHTS (2).png",
                "HIGHLIGHT 3": "static/images/HIGHLIGHTS (3).png",
                "HIGHLIGHT 4": "static/images/HIGHLIGHTS (4).png",
                "HIGHLIGHT 5": "static/images/HIGHLIGHTS (5).png"
                # Add more here if needed
            }
            
            image_path = image_map.get(text)
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Title card image not found for text: '{text}' at path: {image_path}")

            print(f"🎬 Creating image card from: {image_path}")

            # 2. Load the image and convert it into a video clip.
            card = ImageClip(image_path).set_duration(duration)
            
            # 3. CRITICAL ASPECT RATIO FIX: Resize the card to match the source video.
            # This is non-negotiable.
            card = card.resize(newsize=(width, height))
            
            # 4. Set the FPS to match the main video for smooth concatenation.
            card = card.set_fps(fps)
            
            print(f"✅ Image card '{text}' created successfully with size {card.size}")
            return card

        except Exception as e:
            print(f"❌ Image card creation failed: {e}")
            # Fallback to a simple black screen if the image fails for any reason.
            return ColorClip(size=(width, height), color=(0, 0, 0), duration=duration).set_fps(fps)

    def _create_fallback_text_card(self, text: str, width: int, height: int, fps: float, duration: float = 2.0):
        """Fallback method to create text card if image loading fails"""
        try:
            print(f"🎬 Creating fallback text card: '{text}' for {duration}s at {width}x{height}")
            
            # Create a black background
            bg_color = (0, 0, 0)  # Black
            card = ColorClip(size=(width, height), color=bg_color, duration=duration)
            print(f"✅ Background card created: {card.size}")
            
            # Calculate dynamic font size
            font_size = int(width * 1.0)  # 40% of frame width
            print(f"📝 Using font size: {font_size}")
            
            # Create text clip with neon purple color
            try:
                text_clip = TextClip(
                    text,
                    fontsize=font_size,
                    color='white',
                    stroke_color='red',
                    stroke_width=20,
                    font='Montserrat-SemiBold.ttf'
                ).set_position('center').set_duration(duration)
                print(f"✅ Text clip created: {text_clip.size}")
            except Exception as text_error:
                print(f"⚠️ Text clip creation failed: {text_error}")
                # Fallback to default font
                text_clip = TextClip(
                    text,
                    fontsize=font_size,
                    color='white',
                    stroke_color='red',
                    stroke_width=20
                ).set_position('center').set_duration(duration)
                print(f"✅ Text clip created with fallback font: {text_clip.size}")
            
            # CRITICAL: Return proper CompositeVideoClip
            final_card = CompositeVideoClip([card, text_clip])
            print(f"✅ Fallback text card '{text}' created successfully - Final size: {final_card.size}")
            
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
            print(f"❌ Fallback text card creation failed: {e}")
            import traceback
            traceback.print_exc()
            # Final fallback: simple color clip
            return ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)

    def _add_overlays(self, frame, action_text, user_name, pose):
        """
        Add text overlays to video frames with enhanced visibility
        """
        try:
            h, w, _ = frame.shape
            
            # Convert frame to PIL for text rendering
            pil_frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_frame)
            
            # CRITICAL: 3x bigger font sizes and 5x thicker outlines
            label_font_size = max(780, int(w * 0.099))  # 3x bigger (was 260)
            caption_font_size = max(390, int(w * 0.702))  # 3x bigger (was 130)
            
            # Load fonts with 3x bigger sizes
            try:
                label_font = ImageFont.truetype("arial.ttf", label_font_size)
                caption_font = ImageFont.truetype("arial.ttf", caption_font_size)
            except:
                # Fallback to default font
                label_font = ImageFont.load_default()
                caption_font = ImageFont.load_default()
            
            # CRITICAL: 5x thicker black outlines
            label_stroke_width = 20  # 5x thicker (was 4)
            caption_stroke_width = 50  # 5x thicker (was 10)
            
            # Head position detection
            head_pos = self._detect_head_position(frame, pose)
            
            # Draw head pointer label (3x bigger text, 5x thicker outline)
            if head_pos:
                x, y = head_pos
                label_text = f"{user_name}"
                
                # CRITICAL: 5x thicker black outline
                draw.text(
                    (x - 50, y - 100),
                    label_text,
                    font=label_font,
                    fill=self.colors['white'],
                    stroke_width=label_stroke_width,  # 5x thicker
                    stroke_fill=self.colors['black']
                )
                
                # Draw red downward-pointing triangle arrow
                arrow_size = 30
                arrow_points = [
                    (x, y - 50),  # Top point
                    (x - arrow_size//2, y - 50 - arrow_size),  # Bottom left
                    (x + arrow_size//2, y - 50 - arrow_size)   # Bottom right
                ]
                draw.polygon(arrow_points, fill=self.colors['red'])
            
            # Draw action text caption (3x bigger text, 5x thicker outline)
            if action_text:
                # CRITICAL: 3x bigger font size
                font_size = max(390, int(w * 0.702))  # 3x bigger (was 130)
                
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Word wrap for better readability
                words = action_text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    # Estimate text width (rough calculation)
                    estimated_width = len(test_line) * font_size * 0.6
                    
                    if estimated_width > w * 0.8:  # 80% of frame width
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line = test_line
                
                if current_line:
                    lines.append(current_line)
                
                # Calculate total text height
                line_height = font_size * 1.2
                total_text_height = len(lines) * line_height
                
                # CRITICAL: Position at 20% from bottom of screen
                text_y = h - total_text_height - int(h * 0.20)
                
                # Draw each line of wrapped text with 5x thicker outline
                for i, line in enumerate(lines):
                    line_y = text_y + (i * line_height)
                    
                    # CRITICAL: 5x thicker black outline
                    draw.text(
                        (int(w * 0.1), line_y),  # 10% from left edge
                        line,
                        font=font,
                        fill=self.colors['white'],
                        stroke_width=caption_stroke_width,  # 5x thicker
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
        Convert timestamp string to seconds - handle multiple formats
        """
        try:
            print(f"🔍 Parsing timestamp: '{timestamp_str}' (type: {type(timestamp_str)})")
            
            if timestamp_str is None:
                print("⚠️ None timestamp, using 0")
                return 0
                
            # Convert to string if it's a number
            if isinstance(timestamp_str, (int, float)):
                result = float(timestamp_str)
                print(f"✅ Parsed numeric timestamp: {timestamp_str} = {result}s")
                return result
                
            # Convert to string for processing
            timestamp_str = str(timestamp_str).strip()
            
            if not timestamp_str:
                print("⚠️ Empty timestamp, using 0")
                return 0
                
            # Handle MM:SS format
            if ':' in timestamp_str:
                parts = timestamp_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = map(int, parts)
                    result = minutes * 60 + seconds
                    print(f"✅ Parsed MM:SS format: {minutes}:{seconds:02d} = {result}s")
                    return result
                elif len(parts) == 3:
                    hours, minutes, seconds = map(int, parts)
                    result = hours * 3600 + minutes * 60 + seconds
                    print(f"✅ Parsed HH:MM:SS format: {hours}:{minutes:02d}:{seconds:02d} = {result}s")
                    return result
            
            # Handle plain number (seconds)
            try:
                result = float(timestamp_str)
                print(f"✅ Parsed as seconds: {result}s")
                return result
            except ValueError:
                print(f"⚠️ Could not parse timestamp '{timestamp_str}', using 0")
                return 0
                
        except Exception as e:
            print(f"❌ Timestamp parsing error: {e}")
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

    def _create_gap_clip(self, width: int, height: int, fps: float, duration: float = 1.5):
        """Create a silent gap clip between highlights"""
        try:
            # Create a black background clip
            gap_clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
            print(f"✅ Gap clip created: {width}x{height} for {duration}s")
            return gap_clip
        except Exception as e:
            print(f"⚠️ Gap clip creation failed: {e}")
            # Fallback: simple black clip
            return ColorClip(size=(width, height), color=(0, 0, 0), duration=duration) 