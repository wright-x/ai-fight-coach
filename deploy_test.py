#!/usr/bin/env python3
"""
Test script to verify syntax fixes and memory optimizations
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported without syntax errors"""
    try:
        logger.info("Testing imports...")
        
        # Test main module
        import main_simple
        logger.info("✅ main_simple.py imports successfully")
        
        # Test video processor
        from utils.video_processor import VideoProcessor
        logger.info("✅ VideoProcessor imports successfully")
        
        # Test other utils
        from utils.gemini_client import GeminiClient
        from utils.tts_client import TTSClient
        logger.info("✅ All utility modules import successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_memory_optimizations():
    """Test memory optimization features"""
    try:
        logger.info("Testing memory optimizations...")
        
        from utils.video_processor import VideoProcessor
        processor = VideoProcessor()
        
        # Test memory check function
        result = processor._check_memory_usage()
        logger.info(f"✅ Memory check function works: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory optimization test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting deployment tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Memory Optimization Test", test_memory_optimizations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        if test_func():
            logger.info(f"✅ {test_name} PASSED")
            passed += 1
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Ready for deployment.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 