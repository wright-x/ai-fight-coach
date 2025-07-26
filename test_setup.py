"""
Test script to verify AI Fight Coach setup
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.logger import logger
        print("✓ Logger imported successfully")
    except Exception as e:
        print(f"✗ Logger import failed: {e}")
        return False
    
    try:
        from utils.video_processor import VideoProcessor
        print("✓ VideoProcessor imported successfully")
    except Exception as e:
        print(f"✗ VideoProcessor import failed: {e}")
        return False
    
    try:
        from utils.gemini_client import GeminiClient
        print("✓ GeminiClient imported successfully")
    except Exception as e:
        print(f"✗ GeminiClient import failed: {e}")
        return False
    
    try:
        from utils.tts_client import TTSClient
        print("✓ TTSClient imported successfully")
    except Exception as e:
        print(f"✗ TTSClient import failed: {e}")
        return False
    
    return True

def test_directories():
    """Test that all required directories exist."""
    print("\nTesting directories...")
    
    required_dirs = ["uploads", "output", "static", "temp", "debug_logs", "prompts", "utils"]
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ Directory {dir_name} exists")
        else:
            print(f"✗ Directory {dir_name} missing")
            return False
    
    return True

def test_files():
    """Test that all required files exist."""
    print("\nTesting files...")
    
    required_files = [
        "requirements.txt",
        "main.py",
        "prompts/default_prompt.txt",
        "utils/__init__.py",
        "utils/logger.py",
        "utils/video_processor.py",
        "utils/gemini_client.py",
        "utils/tts_client.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ File {file_path} exists")
        else:
            print(f"✗ File {file_path} missing")
            return False
    
    return True

def test_environment():
    """Test environment variables."""
    print("\nTesting environment...")
    
    # Check if API keys are set (but don't validate them)
    google_key = os.getenv('GOOGLE_API_KEY')
    elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
    
    if google_key:
        print("✓ GOOGLE_API_KEY is set")
    else:
        print("⚠ GOOGLE_API_KEY not set (will be required for Gemini)")
    
    if elevenlabs_key:
        print("✓ ELEVENLABS_API_KEY is set")
    else:
        print("⚠ ELEVENLABS_API_KEY not set (will be required for TTS)")
    
    return True

def main():
    """Run all tests."""
    print("AI Fight Coach Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_directories,
        test_files,
        test_environment
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! Application is ready to run.")
        print("\nTo start the server:")
        print("python main.py")
        print("\nOr with uvicorn:")
        print("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 