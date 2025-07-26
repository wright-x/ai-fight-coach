# AI Fight Coach

AI-powered boxing analysis application that provides personalized coaching feedback using video analysis, Gemini AI, and ElevenLabs TTS.

## Features

- **Video Analysis**: Upload boxing videos for AI-powered analysis
- **Frame-by-Frame Processing**: Videos are slowed down 4x for detailed analysis
- **AI Feedback**: Gemini AI provides structured coaching feedback
- **TTS Narration**: ElevenLabs generates natural speech from AI feedback
- **Video Overlays**: Text overlays highlight key moments and feedback
- **Highlights Video**: Automatic creation of highlight reels
- **Complete Pipeline**: End-to-end processing from upload to final video

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### 3. Test Setup

```bash
python test_setup.py
```

### 4. Run the Application

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Upload Video
```
POST /upload
```

Upload a video file for analysis. Supports:
- `video`: Video file (MP4, AVI, MOV, MKV)
- `custom_prompt`: Optional custom analysis prompt
- `voice_id`: Optional ElevenLabs voice ID

### Check Status
```
GET /status/{job_id}
```

Check the status of an analysis job.

### List Voices
```
GET /voices
```

Get available ElevenLabs voices for TTS.

### Download Result
```
GET /static/{filename}
```

Download the final processed video.

## Application Flow

1. **Upload**: User uploads a boxing video
2. **Slow Down**: Video is slowed down 4x for frame-by-frame analysis
3. **AI Analysis**: Gemini AI analyzes the slowed video
4. **Cleanup**: Slowed video is immediately deleted
5. **TTS Generation**: ElevenLabs creates audio from AI feedback
6. **Overlays**: Text overlays are added to the original video
7. **Highlights**: Highlight video is created from key moments
8. **Merge**: Final video with audio is created
9. **Serve**: Video is served via static endpoint
10. **Cleanup**: All temporary files are cleaned up

## Directory Structure

```
ai-fight-coach/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── test_setup.py          # Setup verification script
├── utils/                  # Core utilities
│   ├── logger.py          # Logging utility
│   ├── video_processor.py # Video processing
│   ├── gemini_client.py   # Gemini AI client
│   └── tts_client.py      # ElevenLabs TTS client
├── prompts/               # AI prompts
│   └── default_prompt.txt # Default Gemini prompt
├── uploads/               # Uploaded videos
├── output/                # Intermediate files
├── static/                # Final videos (served)
├── temp/                  # Temporary files
└── debug_logs/            # Application logs
```

## Requirements

- Python 3.11+
- Google Gemini API key
- ElevenLabs API key
- FFmpeg (for video processing)

## Development

The application is built with:
- **FastAPI**: Web framework
- **Google Generative AI**: Video analysis
- **ElevenLabs**: Text-to-speech
- **OpenCV**: Video processing
- **MoviePy**: Video manipulation
- **Pillow**: Image processing for overlays 