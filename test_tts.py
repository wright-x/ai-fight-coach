#!/usr/bin/env python3
"""
Simple unit test for TTS pipeline
"""

import os
import sys
sys.path.append('.')

from utils.tts_client import TTSClient
from utils.video_processor import VideoProcessor

def test_tts_pipeline():
    """Test TTS pipeline with 'Two sentences.' input"""
    try:
        # Initialize TTS client
        tts_client = TTSClient()
        
        # Test data
        test_highlight = {
            'action_required': 'Two sentences. This is the second sentence.'
        }
        
        # Create temp directory
        os.makedirs('output/audio', exist_ok=True)
        
        # Generate TTS
        processor = VideoProcessor()
        processor.tts_client = tts_client
        
        audio_path = processor._generate_highlight_tts(test_highlight, 'output/audio')
        
        if audio_path and os.path.exists(audio_path):
            # Check audio duration
            from moviepy.editor import AudioFileClip
            audio = AudioFileClip(audio_path)
            duration = audio.duration
            audio.close()
            
            # Expected duration: 2 sentences + 1 silence gap = ~2.4s
            expected_duration = 2.4
            tolerance = 0.1
            
            print(f"✅ TTS test passed!")
            print(f"   Audio duration: {duration:.2f}s")
            print(f"   Expected: {expected_duration:.2f}s ± {tolerance}s")
            print(f"   Difference: {abs(duration - expected_duration):.2f}s")
            
            if abs(duration - expected_duration) < tolerance:
                print("✅ Duration within tolerance!")
                return True
            else:
                print("❌ Duration outside tolerance!")
                return False
        else:
            print("❌ No audio file generated")
            return False
            
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

if __name__ == "__main__":
    test_tts_pipeline() 