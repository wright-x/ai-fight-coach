#!/usr/bin/env python3
"""
Debug startup script to test imports and configuration
"""

import os
import sys

def test_imports():
    """Test all imports to see what's failing"""
    print("Testing imports...")
    
    try:
        print("✓ Importing FastAPI...")
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
    except Exception as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    try:
        print("✓ Importing utils...")
        from utils.video_processor import VideoProcessor
        print("✓ VideoProcessor imported successfully")
    except Exception as e:
        print(f"✗ VideoProcessor import failed: {e}")
        return False
    
    try:
        print("✓ Importing GeminiClient...")
        from utils.gemini_client import GeminiClient
        print("✓ GeminiClient imported successfully")
    except Exception as e:
        print(f"✗ GeminiClient import failed: {e}")
        return False
    
    try:
        print("✓ Importing TTSClient...")
        from utils.tts_client import TTSClient
        print("✓ TTSClient imported successfully")
    except Exception as e:
        print(f"✗ TTSClient import failed: {e}")
        return False
    
    try:
        print("✓ Importing UserManager...")
        from user_management import UserManager
        print("✓ UserManager imported successfully")
    except Exception as e:
        print(f"✗ UserManager import failed: {e}")
        return False
    
    try:
        print("✓ Importing email_config...")
        from email_config import SMTP_CONFIG, ADMIN_EMAIL
        print("✓ email_config imported successfully")
    except Exception as e:
        print(f"✗ email_config import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables"""
    print("\nTesting environment variables...")
    
    required_vars = [
        'GOOGLE_API_KEY',
        'ELEVENLABS_API_KEY',
        'SMTP_EMAIL',
        'SMTP_PASSWORD'
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var} is set")
        else:
            print(f"✗ {var} is NOT set")
    
    return True

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = ['static', 'uploads', 'output', 'temp']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name} directory exists")
        else:
            print(f"✗ {dir_name} directory missing")
    
    return True

if __name__ == "__main__":
    print("=== AI Fight Coach Startup Debug ===")
    
    success = True
    success &= test_imports()
    success &= test_environment()
    success &= test_directories()
    
    if success:
        print("\n✅ All tests passed! App should start successfully.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1) 