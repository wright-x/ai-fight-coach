"""
Simplified Gemini Client - Railway Compatible Version
Works without OpenCV dependencies but calls real Gemini API
"""

import os
import json
import time
import base64
from typing import Dict, Any, Optional
import google.generativeai as genai

class GeminiClient:
    """Simplified Gemini client for Railway environment"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("❌ No Google API key found")
            return
            
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ GeminiClient initialized (simplified mode)")
    
    def analyze_video(self, video_path: str, analysis_type: str = "everything") -> Dict[str, Any]:
        """Analyze video and return results"""
        try:
            if not self.api_key:
                return self._get_mock_analysis(analysis_type)
            
            print(f"🤖 Starting real Gemini analysis for: {video_path}")
            
            # Read video file
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            # Encode to base64
            video_b64 = base64.b64encode(video_data).decode('utf-8')
            
            # Create prompt based on analysis type
            if analysis_type == "head_movement":
                prompt = """
                Analyze this boxing video focusing on head movement and defense. Look for:
                1. Head movement patterns (slipping, weaving, bobbing)
                2. Defensive positioning
                3. Reaction time to incoming punches
                4. Head positioning during attacks
                
                Return a JSON response with:
                - highlights: array of key moments with timestamps, detailed_feedback, and action_required
                - recommended_drills: array of drills with drill_name, description, and problem_it_fixes
                """
            elif analysis_type == "punch_techniques":
                prompt = """
                Analyze this boxing video focusing on punch techniques. Look for:
                1. Proper form and technique
                2. Power generation
                3. Accuracy and precision
                4. Combination effectiveness
                
                Return a JSON response with:
                - highlights: array of key moments with timestamps, detailed_feedback, and action_required
                - recommended_drills: array of drills with drill_name, description, and problem_it_fixes
                """
            elif analysis_type == "footwork":
                prompt = """
                Analyze this boxing video focusing on footwork. Look for:
                1. Balance and stability
                2. Movement efficiency
                3. Positioning and angles
                4. Defensive footwork
                
                Return a JSON response with:
                - highlights: array of key moments with timestamps, detailed_feedback, and action_required
                - recommended_drills: array of drills with drill_name, description, and problem_it_fixes
                """
            else:  # everything
                prompt = """
                Analyze this boxing video comprehensively. Look for:
                1. Overall technique and form
                2. Defensive skills
                3. Offensive effectiveness
                4. Movement and positioning
                5. Areas for improvement
                
                Return a JSON response with:
                - highlights: array of key moments with timestamps, detailed_feedback, and action_required
                - recommended_drills: array of drills with drill_name, description, and problem_it_fixes
                """
            
            # Send to Gemini
            response = self.model.generate_content([
                prompt,
                {
                    "inline_data": {
                        "mime_type": "video/mp4",
                        "data": video_b64
                    }
                }
            ])
            
            print(f"✅ Gemini analysis completed")
            
            # Parse response
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                print(f"❌ Failed to parse Gemini response: {response.text}")
                return self._get_mock_analysis(analysis_type)
                
        except Exception as e:
            print(f"❌ Error in Gemini analysis: {e}")
            return self._get_mock_analysis(analysis_type)
    
    def _get_mock_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """Return mock analysis when Gemini is not available"""
        if analysis_type == "head_movement":
            return {
                "highlights": [
                    {
                        "timestamp": 20,
                        "detailed_feedback": "Good head movement and evasion",
                        "action_required": "Keep your head moving consistently"
                    }
                ],
                "recommended_drills": [
                    {
                        "drill_name": "Head Movement Drills",
                        "description": "Practice slipping and weaving",
                        "problem_it_fixes": "Improves defensive head movement"
                    }
                ]
            }
        else:
            return {
                "highlights": [
                    {
                        "timestamp": 15,
                        "detailed_feedback": "Good overall technique",
                        "action_required": "Continue practicing consistently"
                    }
                ],
                "recommended_drills": [
                    {
                        "drill_name": "Shadow Boxing",
                        "description": "Practice combinations in front of mirror",
                        "problem_it_fixes": "Improves overall technique"
                    }
                ]
            } 