# AI Fight Coach - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Core Components](#core-components)
5. [API Endpoints](#api-endpoints)
6. [Frontend Components](#frontend-components)
7. [Configuration](#configuration)
8. [Deployment](#deployment)
9. [Changelog](#changelog)

## Project Overview

AI Fight Coach is a web application that provides AI-powered boxing analysis using computer vision, natural language processing, and video processing. The system analyzes uploaded boxing videos to identify technique issues, generate highlights, provide personalized feedback, and recommend specific drills and YouTube videos for improvement.

### Key Features
- **AI Video Analysis**: Uses Google Gemini AI for technical boxing analysis
- **Highlight Detection**: Identifies key moments and technique breakdowns
- **Head Tracking**: OpenCV-based pose detection for fighter identification
- **Personalized Feedback**: Custom analysis based on individual performance
- **Drill Recommendations**: Specific exercises to address identified problems
- **YouTube Integration**: Curated video recommendations for skill improvement
- **Auto-Cleanup**: Automatic file deletion after 15 minutes
- **User Registration**: Lead generation through name/email collection

## Architecture

The application follows a modern web architecture with:
- **Backend**: FastAPI (Python) for API endpoints and video processing
- **Frontend**: HTML/CSS/JavaScript with modern UI design
- **AI Services**: Google Gemini for analysis, ElevenLabs for TTS (optional)
- **Video Processing**: MoviePy, OpenCV, MediaPipe for video manipulation
- **Storage**: Local file system with automatic cleanup

## File Structure

```
ai-fight-coach/
├── main.py                 # FastAPI application and API endpoints
├── start_server.py         # Server startup with environment variables
├── static/                 # Frontend files
│   ├── index.html         # Main application interface
│   └── register.html      # User registration page
├── utils/                  # Core utility modules
│   ├── __init__.py
│   ├── video_processor.py # Video processing and manipulation
│   ├── gemini_client.py   # Google Gemini AI integration
│   ├── tts_client.py      # ElevenLabs TTS integration
│   └── logger.py          # Logging configuration
├── prompts/               # AI prompt templates
│   └── default_prompt.txt # Main analysis prompt
├── uploads/               # Temporary uploaded videos
├── output/                # Processed video outputs
├── temp/                  # Temporary processing files
├── debug_logs/            # Debug and analysis logs
├── requirements.txt       # Python dependencies
├── README.md             # Basic project information
├── QUICK_START.md        # Quick setup guide
└── DOCUMENTATION.md      # This file
```

## Core Components

### 1. main.py - FastAPI Application

**Purpose**: Main application server with API endpoints and background processing.

**Key Functions**:
- `startup_event()`: Initialize application and create directories
- `upload_video()`: Handle video uploads and start analysis jobs
- `get_status()`: Return job status and results
- `process_video_analysis()`: Background task for video processing
- `schedule_file_deletion()`: Schedule files for automatic cleanup
- `cleanup_old_files()`: Remove expired files
- `cleanup_worker()`: Background worker for file cleanup

**Key Features**:
- Job management with unique IDs
- Background task processing
- Automatic file cleanup (15-minute expiration)
- Error handling and logging
- User registration integration

### 2. utils/video_processor.py - Video Processing Engine

**Purpose**: Handle all video manipulation, overlay creation, and highlight generation.

**Key Functions**:
- `__init__()`: Initialize MediaPipe pose detection and temp directory
- `get_video_info()`: Extract video metadata (fps, duration, dimensions)
- `slow_down_video()`: Create slowed video for frame-by-frame analysis
- `create_highlight_video()`: Generate highlights video with overlays and head tracking
- `add_overlays_to_video()`: Add text overlays and pose detection to videos
- `_draw_head_tracking()`: Draw fighter pointer and name using pose detection
- `merge_video_audio()`: Combine video with audio tracks
- `cleanup_temp_files()`: Remove temporary processing files

**Key Features**:
- OpenCV-based head tracking with MediaPipe
- MoviePy video manipulation
- Custom overlay system with text rendering
- Slow-motion highlight generation
- Fighter name display with visual pointers

### 3. utils/gemini_client.py - AI Analysis Engine

**Purpose**: Interface with Google Gemini AI for video analysis and feedback generation.

**Key Functions**:
- `__init__()`: Initialize Gemini client with API key
- `analyze_video()`: Send video to Gemini for analysis
- `validate_analysis_result()`: Validate AI response structure
- `get_analysis_summary()`: Generate summary statistics
- `_prepare_video_for_analysis()`: Extract frames and prepare for AI
- `_log_request_response()`: Debug logging for AI interactions

**Key Features**:
- Structured JSON response validation
- Frame extraction for video analysis
- Error handling and retry logic
- Comprehensive logging for debugging
- Support for custom prompts

### 4. utils/tts_client.py - Text-to-Speech Engine

**Purpose**: Generate audio feedback using ElevenLabs TTS service.

**Key Functions**:
- `__init__()`: Initialize ElevenLabs client and load voices
- `generate_full_analysis_audio()`: Create complete audio feedback
- `_prepare_full_analysis_text()`: Format text for TTS with pauses
- `list_voices()`: Get available TTS voices

**Key Features**:
- Multiple voice options
- Pause insertion for natural speech
- Audio file generation and management
- Error handling for TTS failures

### 5. utils/logger.py - Logging System

**Purpose**: Centralized logging configuration for the entire application.

**Key Functions**:
- Configure logging levels and formats
- File-based logging with rotation
- Console and file output
- Debug log management

## API Endpoints

### POST /upload
**Purpose**: Upload video for analysis
**Parameters**: 
- `video`: Video file (multipart/form-data)
- `fighter_name`: Optional fighter name (default: "FIGHTER")
**Response**: Job ID and status

### GET /status/{job_id}
**Purpose**: Get analysis job status and results
**Parameters**: `job_id`: Unique job identifier
**Response**: Job status, video URL, and analysis results

### GET /register
**Purpose**: Serve user registration page
**Response**: Registration HTML page

### GET /main
**Purpose**: Serve main application page
**Response**: Main application HTML page

### GET /
**Purpose**: Redirect to registration page
**Response**: Redirect to /register

## Frontend Components

### static/index.html - Main Application Interface

**Key Features**:
- Modern, responsive design with gradient backgrounds
- File upload with drag-and-drop support
- Real-time progress bar with custom timing
- Video player with download functionality
- Analysis results display with formatting
- User registration integration
- Error handling and user feedback

**JavaScript Functions**:
- `startAnalysis()`: Handle video upload and processing
- `pollStatus()`: Check job status and update UI
- `updateProgress()`: Animate progress bar with custom timing
- `displayAnalysis()`: Render analysis results with formatting
- `showResult()`: Display completed video and results
- `showError()`: Handle and display errors

### static/register.html - User Registration

**Key Features**:
- Clean registration form with validation
- Local storage for user data persistence
- Automatic redirect to main application
- Feature showcase and benefits display

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Google Gemini API key
- `ELEVENLABS_API_KEY`: ElevenLabs TTS API key (optional)

### File Paths
- Uploads: `uploads/` directory
- Outputs: `static/` directory for web access
- Temp files: `temp/` directory
- Logs: `debug_logs/` directory

### Auto-Cleanup Settings
- File retention: 15 minutes
- Cleanup interval: 60 seconds
- Affected files: All uploaded and processed videos

## Deployment

### Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- Google Gemini API key
- ElevenLabs API key (optional)

### Installation
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables
4. Run: `python start_server.py`

### Production Considerations
- Use proper WSGI server (Gunicorn)
- Configure reverse proxy (Nginx)
- Set up SSL certificates
- Monitor disk space for video storage
- Implement proper error monitoring

## Changelog

### Version 1.0.0 - MVP Release (July 26, 2025)

#### Core Features Implemented
- **AI Video Analysis**: Integrated Google Gemini AI for boxing technique analysis
- **Highlight Detection**: Automatic identification of key moments and technique breakdowns
- **Video Processing Pipeline**: Complete video upload, processing, and output generation
- **User Registration System**: Lead generation through name/email collection
- **Modern UI/UX**: Responsive design with gradient backgrounds and smooth animations

#### Technical Improvements
- **Loading Bar Implementation**: Custom 3-minute progress bar with realistic timing simulation
- **File Management**: Automatic cleanup system with 15-minute file retention
- **Error Handling**: Comprehensive error handling and user feedback
- **Logging System**: Detailed logging for debugging and monitoring

#### Analysis Features
- **Structured Feedback**: Highlights with timestamps, detailed feedback, and action items
- **Recommended Drills**: Specific exercises to address identified problems
- **YouTube Integration**: Curated video recommendations for skill improvement
- **Head Tracking**: OpenCV-based pose detection for fighter identification

#### UI/UX Enhancements
- **Registration Flow**: Separate registration page with feature showcase
- **Progress Tracking**: Real-time progress updates with custom timing
- **Result Display**: Formatted analysis results with emojis and styling
- **Video Player**: Integrated video player with download functionality

#### Backend Architecture
- **FastAPI Framework**: Modern async web framework for high performance
- **Background Processing**: Non-blocking video analysis with job management
- **Modular Design**: Separated concerns with dedicated utility modules
- **Auto-Cleanup**: Background worker for automatic file management

#### Recent Fixes and Improvements
- **Highlight Text Fix**: Resolved issue where all highlights showed identical text
- **Loading Bar Timing**: Implemented exact timing requirements (30s to 50%, 60s to 75%, 15s to 90%, 60s to 100%)
- **YouTube Verification**: Enhanced prompt to ensure only real, existing videos are recommended
- **Drill Recommendations**: Replaced generic "Areas for Improvement" with specific "Recommended Drills"
- **Schema Validation**: Comprehensive validation for all AI response structures

#### Performance Optimizations
- **Video Processing**: Optimized video manipulation with MoviePy and OpenCV
- **Memory Management**: Efficient file handling and cleanup
- **Response Time**: Fast API responses with background processing
- **User Experience**: Smooth animations and responsive interface

#### Security and Reliability
- **Input Validation**: Comprehensive validation of uploaded files and API responses
- **Error Recovery**: Graceful handling of processing failures
- **File Security**: Automatic cleanup prevents disk space issues
- **API Key Management**: Secure handling of external service credentials

---

**Next Steps for Future Versions:**
- Real-time progress tracking based on actual processing time
- Advanced pose detection and movement analysis
- Integration with additional AI models for enhanced analysis
- User dashboard with analysis history
- Social features and progress sharing
- Mobile application development 