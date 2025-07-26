"""
Gemini AI client for video analysis in AI Fight Coach application.
Handles video analysis requests with comprehensive logging and error handling.
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from .logger import logger


class GeminiClient:
    """Client for interacting with Google's Gemini AI for video analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("GeminiClient initialized successfully")
    
    def _encode_video_to_base64(self, video_path: str) -> str:
        """Encode video file to base64 string."""
        try:
            with open(video_path, 'rb') as video_file:
                video_data = video_file.read()
                encoded_video = base64.b64encode(video_data).decode('utf-8')
                logger.debug(f"Video encoded to base64: {len(encoded_video)} characters")
                return encoded_video
        except Exception as e:
            logger.error(f"Error encoding video to base64: {e}")
            raise
    
    def _load_prompt_template(self, prompt_path: str = "prompts/default_prompt.txt") -> str:
        """Load prompt template from file."""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                logger.debug(f"Loaded prompt template from {prompt_path}")
                return prompt
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            raise
    
    def _generate_final_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """
        Generate the final prompt by combining default prompt with custom instructions.
        
        Args:
            custom_prompt: Optional custom instructions from user
            
        Returns:
            Final prompt to use for video analysis
        """
        try:
            default_prompt = self._load_prompt_template()
            
            if not custom_prompt:
                logger.info("No custom prompt provided, using default prompt")
                return default_prompt
            
            # Create a prompt to modify the default prompt based on custom instructions
            modification_prompt = f"""
You are a prompt engineering expert. I have a default prompt for boxing video analysis that generates JSON output. 

DEFAULT PROMPT:
{default_prompt}

USER CUSTOM INSTRUCTIONS:
{custom_prompt}

TASK: Modify the default prompt to incorporate the user's custom instructions while maintaining the exact JSON schema and format requirements. The final prompt must still generate valid JSON with the same structure.

IMPORTANT: 
- Keep the JSON schema exactly the same
- Keep all the "CRITICAL" and "FINAL INSTRUCTION" parts about JSON-only output
- Only modify the analysis instructions and rules based on user input
- Return ONLY the modified prompt, no explanations

MODIFIED PROMPT:
"""
            
            logger.info("Generating modified prompt based on custom instructions")
            
            # Get modified prompt from Gemini
            response = self.model.generate_content(modification_prompt)
            final_prompt = response.text.strip()
            
            logger.info("Successfully generated modified prompt")
            return final_prompt
            
        except Exception as e:
            logger.error(f"Error generating final prompt: {e}")
            # Fallback to default prompt
            logger.info("Falling back to default prompt")
            return self._load_prompt_template()

    def _log_request_response(self, type: str, data: dict):
        """
        Log request and response data for debugging.
        
        Args:
            type: Either "request" or "response"
            data: Data to log
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_data = {
                "type": type,
                "timestamp": timestamp,
                "data": data
            }
            logger.info(f"Gemini {type}: {json.dumps(log_data, indent=2)}")
        except Exception as e:
            logger.error(f"Error logging {type}: {e}")

    def _prepare_video_for_analysis(self, video_path: str) -> list:
        """
        Prepare video for analysis by encoding it to base64.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List containing the encoded video data
        """
        try:
            # Encode video to base64
            encoded_video = self._encode_video_to_base64(video_path)
            
            # Return as a list with the video data
            return [{
                "inline_data": {
                    "mime_type": "video/mp4",
                    "data": encoded_video
                }
            }]
        except Exception as e:
            logger.error(f"Error preparing video for analysis: {e}")
            raise

    def analyze_video(self, video_path: str, prompt_file: str = "prompts/default_prompt.txt") -> dict:
        """
        Analyze a video using Gemini AI and return structured feedback.
        
        Args:
            video_path: Path to the video file
            prompt_file: Path to the prompt file to use for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting video analysis: {video_path}")
            
            # Load the prompt from file
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                logger.info(f"Loaded prompt from: {prompt_file}")
            except Exception as e:
                logger.error(f"Error loading prompt file {prompt_file}: {e}")
                # Fallback to default prompt
                with open("prompts/default_prompt.txt", 'r', encoding='utf-8') as f:
                    prompt = f.read()
                logger.info("Using fallback default prompt")
            
            # Prepare video for analysis
            video_data = self._prepare_video_for_analysis(video_path)
            
            # Log request for debugging
            self._log_request_response("request", {
                "video_path": video_path,
                "prompt_file": prompt_file,
                "prompt": prompt
            })
            
            # Send to Gemini
            logger.info("Sending request to Gemini AI...")
            response = self.model.generate_content([prompt, *video_data])
            
            # Log response for debugging
            self._log_request_response("response", {
                "response_text": response.text,
                "response_parts": len(response.parts) if hasattr(response, 'parts') else 0
            })
            
            logger.info("Received response from Gemini AI")
            
            # Clean up response text - remove markdown code blocks if present
            cleaned_response = response.text.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]  # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON response
            try:
                result = json.loads(cleaned_response)
                logger.info("Successfully parsed JSON response from Gemini")
                
                # Validate the result
                if not self.validate_analysis_result(result):
                    raise ValueError("Invalid analysis result structure")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response.text}")
                logger.error(f"Cleaned response: {cleaned_response}")
                raise ValueError(f"Invalid JSON response from Gemini: {e}")
                
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            raise
    
    def validate_analysis_result(self, result: dict) -> bool:
        """Validate the structure of the analysis result from Gemini."""
        try:
            if not isinstance(result, dict):
                logger.error("Analysis result must be a dictionary")
                return False
            
            if 'highlights' not in result:
                logger.error("Missing 'highlights' key in analysis result")
                return False
            
            if not isinstance(result['highlights'], list):
                logger.error("highlights must be a list")
                return False
            
            for highlight in result['highlights']:
                if not isinstance(highlight, dict):
                    logger.error("Each highlight item must be a dict")
                    return False
                if 'timestamp' not in highlight or 'detailed_feedback' not in highlight:
                    logger.error("Highlight items must have 'timestamp' and 'detailed_feedback' keys")
                    return False
                if 'action_required' not in highlight:
                    logger.warning("highlight missing 'action_required' field")
            
            # Validate recommended_drills if present
            if 'recommended_drills' in result:
                if not isinstance(result['recommended_drills'], list):
                    logger.error("recommended_drills must be a list")
                    return False
                
                for drill in result['recommended_drills']:
                    if not isinstance(drill, dict):
                        logger.error("Each recommended_drill item must be a dict")
                        return False
                    if 'drill_name' not in drill or 'description' not in drill or 'problem_it_fixes' not in drill:
                        logger.error("recommended_drill items must have 'drill_name', 'description', and 'problem_it_fixes' keys")
                        return False
            
            # Validate youtube_recommendations if present
            if 'youtube_recommendations' in result:
                if not isinstance(result['youtube_recommendations'], list):
                    logger.error("youtube_recommendations must be a list")
                    return False
                
                for rec in result['youtube_recommendations']:
                    if not isinstance(rec, dict):
                        logger.error("Each youtube_recommendation item must be a dict")
                        return False
                    if 'title' not in rec or 'url' not in rec or 'problem_solved' not in rec:
                        logger.error("youtube_recommendation items must have 'title', 'url', and 'problem_solved' keys")
                        return False
                    # Basic URL validation
                    if not rec['url'].startswith('https://www.youtube.com/') and not rec['url'].startswith('https://youtu.be/'):
                        logger.warning(f"Invalid YouTube URL format: {rec['url']}")
            
            logger.info("Analysis result validation passed")
            return True
        except Exception as e:
            logger.error(f"Error validating analysis result: {e}")
            return False
    
    def get_analysis_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the analysis result.
        
        Args:
            result: Analysis result from Gemini
            
        Returns:
            Summary dictionary
        """
        try:
            # Only expect and validate highlights in the result
            # Remove all session_feedback and drills logic
            if not isinstance(result, dict):
                logger.error("Analysis result must be a dictionary")
                return {}
            
            if 'highlights' not in result:
                logger.error("Missing 'highlights' key in analysis result")
                return {}
            
            if not isinstance(result['highlights'], list):
                logger.error("highlights must be a list")
                return {}
            
            summary = {
                "total_highlights": len(result.get('highlights', [])),
                "timestamp_range": {
                    "min": min([f.get('timestamp', 0) for f in result.get('highlights', [])], default=0),
                    "max": max([f.get('timestamp', 0) for f in result.get('highlights', [])], default=0)
                }
            }
            
            logger.info(f"Analysis summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating analysis summary: {e}")
            return {} 