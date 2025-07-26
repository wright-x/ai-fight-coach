#!/usr/bin/env python3
"""
AI Fight Coach Server Launcher
Sets environment variables and starts the FastAPI server
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Set environment variables
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyDsJRnbA3GZckLE83mK2yA2bIYMmungtQA'
    os.environ['ELEVENLABS_API_KEY'] = 'sk_cce495b4c5d2cf5661ad1645be482965997e6f0fe258588d'
    
    print("🚀 Starting AI Fight Coach Server...")
    print(f"📍 Google API Key: {os.environ['GOOGLE_API_KEY'][:20]}...")
    print(f"🎤 ElevenLabs API Key: {os.environ['ELEVENLABS_API_KEY'][:20]}...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📝 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 