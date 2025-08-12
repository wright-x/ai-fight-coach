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
        # Support both GEMINI_API_KEY and GOOGLE_API_KEY
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("❌ No Gemini API key found (GEMINI_API_KEY or GOOGLE_API_KEY)")
            return
            
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use gemini-2.5-flash for better performance
        model_name = os.getenv("GEMINI_MODEL_COACH", "gemini-2.5-flash")
        self.model = genai.GenerativeModel(model_name)
        print(f"✅ GeminiClient initialized with {model_name}")
    
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
            prompt_file_map = {
                "everything": "prompts/everything_prompt.txt",
                "head_movement": "prompts/head_movement_prompt.txt",
                "punch_techniques": "prompts/punch_techniques_prompt.txt",
                "footwork": "prompts/footwork_prompt.txt"
            }
            
            prompt_file = prompt_file_map.get(analysis_type, "prompts/default_prompt.txt")
            
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
                      "short_text": "Tuck your chin and keep your guard up",
                      "long_text": "At 15 seconds, I observed that your guard was slightly lowered and your chin was exposed. This creates a vulnerability that an opponent could exploit. You should maintain a tight guard position with your hands protecting your face at all times, and keep your chin tucked down to minimize exposure to head shots.",
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
                
                CRITICAL: For each highlight, provide both:
                - "short_text": A concise, punchy summary for TTS and on-screen captions (max 50 characters)
                - "long_text": A detailed, paragraph-form explanation for the web results page
                
                Do not include any text before or after the JSON. Return ONLY the JSON object.
                """
            
            # Load video database and add to prompt
            try:
                from video_database import VIDEO_DATABASE
                video_db_text = f"""
                
                VIDEO DATABASE - ONLY USE THESE VIDEOS:
                
                Available categories: {list(VIDEO_DATABASE.keys())}
                
                For analysis_type "{analysis_type}", use videos from these categories:
                - "everything" - for general boxing fundamentals
                - "head_movement" - for defense and head movement issues  
                - "punch_techniques" - for punching technique problems
                - "footwork" - for footwork and positioning issues
                
                Available videos by category:
                """
                
                for category, videos in VIDEO_DATABASE.items():
                    video_db_text += f"\n{category.upper()} VIDEOS:\n"
                    for i, video in enumerate(videos[:10], 1):  # Show first 10 videos per category
                        video_db_text += f"{i}. {video['title']} - {video['url']}\n"
                
                prompt += video_db_text
                print(f"📚 Added video database to prompt")
                
            except ImportError as e:
                print(f"⚠️ Could not load video database: {e}")
            
            # Send to Gemini
            response = self.model.generate_content([
                prompt,
                {"inline_data": {"mime_type": "video/mp4", "data": video_b64}}
            ])
            
            print(f"✅ Gemini analysis completed")
            
            try:
                response_text = response.text.strip()
                print(f"📄 Raw response length: {len(response_text)} characters")
                print(f"📄 Response preview: {response_text[:200]}...")
                
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
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
                        "timestamp": "00:20",
                        "short_text": "Keep your head moving",
                        "long_text": "At 20 seconds, I noticed your head movement was too static. You were standing still for too long, making you an easy target. Good boxers keep their head moving constantly to avoid punches and create angles for counter-attacks.",
                        "action_required": "Practice head movement drills daily"
                    }
                ],
                "recommended_drills": [
                    {
                        "drill_name": "Head Movement Drills",
                        "description": "Practice slipping and weaving",
                        "problem_it_fixes": "Static head positioning"
                    }
                ],
                "youtube_recommendations": [
                    {
                        "title": "Boxing Head Movement Masterclass",
                        "url": "https://www.youtube.com/watch?v=example1",
                        "problem_solved": "Improves head movement and evasion"
                    }
                ]
            }
        else:
            return {
                "highlights": [
                    {
                        "timestamp": "00:15",
                        "short_text": "Tuck your chin and keep your guard up",
                        "long_text": "At 15 seconds, I observed that your guard was slightly lowered and your chin was exposed. This creates a vulnerability that an opponent could exploit. You should maintain a tight guard position with your hands protecting your face at all times.",
                        "action_required": "Continue practicing guard position"
                    }
                ],
                "recommended_drills": [
                    {
                        "drill_name": "Guard Position Drill",
                        "description": "Practice maintaining proper guard",
                        "problem_it_fixes": "Low guard and exposed chin"
                    }
                ],
                "youtube_recommendations": [
                    {
                        "title": "Boxing Guard Fundamentals",
                        "url": "https://www.youtube.com/watch?v=example2",
                        "problem_solved": "Improves guard positioning and defense"
                    }
                ]
            } 