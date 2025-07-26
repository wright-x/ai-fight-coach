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
            
            # Read prompt from file based on analysis type
            prompt_file = f"prompts/{analysis_type}_prompt.txt"
            if analysis_type == "everything":
                prompt_file = "prompts/everything_prompt.txt"
            
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                print(f"📄 Loaded prompt from: {prompt_file}")
            except FileNotFoundError:
                print(f"⚠️ Prompt file not found: {prompt_file}, using default")
                # Fallback to default prompt
                prompt = """
                Analyze this boxing video comprehensively. Look for:
                1. Overall technique and form
                2. Defensive skills
                3. Offensive effectiveness
                4. Movement and positioning
                5. Areas for improvement
                
                IMPORTANT: Return ONLY a valid JSON object with this exact structure:
                {
                  "highlights": [
                    {
                      "timestamp": "00:15",
                      "detailed_feedback": "Description of what was observed",
                      "action_required": "What the fighter should do to improve"
                    }
                  ],
                  "recommended_drills": [
                    {
                      "drill_name": "Name of the drill",
                      "description": "How to perform the drill",
                      "problem_it_fixes": "What problem this drill addresses"
                    }
                  ],
                  "youtube_recommendations": [
                    {
                      "title": "Title of the YouTube video",
                      "url": "https://www.youtube.com/watch?v=VIDEO_ID",
                      "problem_solved": "What problem this video helps solve"
                    }
                  ]
                }
                
                Do not include any text before or after the JSON. Return ONLY the JSON object.
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
                # Clean up Gemini response - remove markdown code blocks and extract JSON
                response_text = response.text.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.startswith('```'):
                    response_text = response_text[3:]  # Remove ```
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove ```
                
                # Find JSON object in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_text = response_text[start_idx:end_idx]
                    result = json.loads(json_text)
                    print(f"✅ Successfully parsed Gemini response")
                    return result
                else:
                    print(f"❌ No JSON object found in response: {response_text}")
                    return self._get_mock_analysis(analysis_type)
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse Gemini response: {response.text}")
                print(f"📋 JSON Error: {e}")
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