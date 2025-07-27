"""
AI Boxing Analysis Video Processor - Basic Version
No video processing dependencies - just file operations and text analysis
"""

import os
import json
import shutil
import tempfile
import time
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

class VideoProcessorBasic:
    """Basic video processor for boxing analysis without video processing libraries"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        print("✅ VideoProcessorBasic initialized (no video processing libraries)")
    
    def create_highlight_video(self, video_path: str, highlights: List[Dict], output_path: str, user_name: str = "FIGHTER") -> str:
        """Create basic boxing analysis - just copy video with metadata"""
        try:
            print(f"🎬 Creating basic boxing analysis: {video_path}")
            print(f"📊 Number of highlights: {len(highlights)}")
            print(f"👤 User: {user_name}")
            
            # Simply copy the original video
            shutil.copy2(video_path, output_path)
            
            # Create analysis metadata file
            metadata_path = output_path.replace('.mp4', '_analysis.json')
            metadata = {
                'user_name': user_name,
                'analysis_date': datetime.now().isoformat(),
                'original_video': video_path,
                'highlights_count': len(highlights),
                'highlights': highlights,
                'processing_mode': 'basic',
                'message': 'Video processing libraries not available. Original video preserved with analysis metadata.'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Basic analysis created: {output_path}")
            print(f"📋 Analysis metadata: {metadata_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error in basic video processor: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            # Ultimate fallback - just copy original
            shutil.copy2(video_path, output_path)
            return output_path
    
    def add_audio_to_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Basic audio handling - just copy video if audio processing fails"""
        try:
            if os.path.exists(audio_path):
                print(f"📁 Audio file found: {audio_path}")
                # Just copy video for now - no audio processing
                shutil.copy2(video_path, output_path)
                print(f"✅ Video copied (audio processing not available): {output_path}")
                return output_path
            else:
                print(f"⚠️ Audio file not found: {audio_path}")
                shutil.copy2(video_path, output_path)
                return output_path
                
        except Exception as e:
            print(f"❌ Error in basic audio handling: {e}")
            shutil.copy2(video_path, output_path)
            return output_path
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get basic video information without processing libraries"""
        try:
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                file_stats = os.stat(video_path)
                
                return {
                    'file_path': video_path,
                    'file_size_bytes': file_size,
                    'file_size_mb': round(file_size / (1024 * 1024), 2),
                    'created_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'processing_mode': 'basic',
                    'message': 'Detailed video properties require processing libraries'
                }
            else:
                return {'error': 'Video file not found'}
                
        except Exception as e:
            return {'error': f'Could not get video info: {e}'} 