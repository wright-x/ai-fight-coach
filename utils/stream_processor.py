"""
Real-time video analysis processor for live streaming
Handles frame analysis, pose detection, and feedback generation
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import base64
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import io
from PIL import Image

logger = logging.getLogger(__name__)

class StreamProcessor:
    """Real-time video analysis for live streaming"""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between speed and accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Analysis state
        self.frame_count = 0
        self.last_analysis_time = datetime.now()
        self.pose_history = []  # Store last 30 poses for pattern analysis
        self.current_errors = []  # Current technique errors
        self.feedback_history = []  # Recent feedback given
        
        # Analysis parameters
        self.analysis_frequency = 1.0  # Analyze every second
        self.feedback_cooldown = 10.0  # 10 seconds between feedback
        self.pose_history_size = 30  # Keep 30 poses in history
        
        logger.info("✅ StreamProcessor initialized")

    def process_frame(self, frame_data: str) -> Dict:
        """
        Process a single frame from the video stream
        Returns analysis results and any immediate feedback
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            self.frame_count += 1
            current_time = datetime.now()
            
            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            analysis_result = {
                'frame_count': self.frame_count,
                'timestamp': current_time.isoformat(),
                'pose_detected': False,
                'errors': [],
                'should_give_feedback': False
            }
            
            if results.pose_landmarks:
                analysis_result['pose_detected'] = True
                
                # Extract key pose points
                pose_data = self._extract_pose_data(results.pose_landmarks)
                
                # Add to pose history
                self.pose_history.append({
                    'timestamp': current_time,
                    'pose_data': pose_data
                })
                
                # Keep only recent poses
                if len(self.pose_history) > self.pose_history_size:
                    self.pose_history.pop(0)
                
                # Analyze technique if we have enough data
                if len(self.pose_history) >= 5:  # Need at least 5 poses
                    errors = self._analyze_technique()
                    analysis_result['errors'] = errors
                    
                    # Check if we should give feedback
                    time_since_last_feedback = (current_time - self.last_analysis_time).total_seconds()
                    if time_since_last_feedback >= self.feedback_cooldown and errors:
                        analysis_result['should_give_feedback'] = True
                        self.last_analysis_time = current_time
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {
                'frame_count': self.frame_count,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def _extract_pose_data(self, landmarks) -> Dict:
        """Extract key pose points for analysis"""
        # Key landmarks for boxing analysis
        key_points = {
            'nose': landmarks.landmark[0],
            'left_shoulder': landmarks.landmark[11],
            'right_shoulder': landmarks.landmark[12],
            'left_elbow': landmarks.landmark[13],
            'right_elbow': landmarks.landmark[14],
            'left_wrist': landmarks.landmark[15],
            'right_wrist': landmarks.landmark[16],
            'left_hip': landmarks.landmark[23],
            'right_hip': landmarks.landmark[24],
            'left_knee': landmarks.landmark[25],
            'right_knee': landmarks.landmark[26],
            'left_ankle': landmarks.landmark[27],
            'right_ankle': landmarks.landmark[28]
        }
        
        # Convert to x, y coordinates
        pose_data = {}
        for name, landmark in key_points.items():
            pose_data[name] = {
                'x': landmark.x,
                'y': landmark.y,
                'visibility': landmark.visibility
            }
        
        return pose_data

    def _analyze_technique(self) -> List[str]:
        """
        Analyze recent poses for technique errors
        Returns list of current errors
        """
        errors = []
        
        if len(self.pose_history) < 5:
            return errors
        
        # Get latest pose
        latest_pose = self.pose_history[-1]['pose_data']
        
        # Check guard position
        guard_error = self._check_guard_position(latest_pose)
        if guard_error:
            errors.append(guard_error)
        
        # Check stance
        stance_error = self._check_stance(latest_pose)
        if stance_error:
            errors.append(stance_error)
        
        # Check head position
        head_error = self._check_head_position(latest_pose)
        if head_error:
            errors.append(head_error)
        
        # Check movement patterns (requires history)
        movement_error = self._check_movement_patterns()
        if movement_error:
            errors.append(movement_error)
        
        return errors

    def _check_guard_position(self, pose_data: Dict) -> Optional[str]:
        """Check if hands are in proper guard position"""
        try:
            left_wrist = pose_data['left_wrist']
            right_wrist = pose_data['right_wrist']
            nose = pose_data['nose']
            
            # Hands should be roughly at face level for guard
            face_level = nose['y']
            left_hand_level = left_wrist['y']
            right_hand_level = right_wrist['y']
            
            # Allow some tolerance
            tolerance = 0.15
            
            if abs(left_hand_level - face_level) > tolerance or abs(right_hand_level - face_level) > tolerance:
                return "Keep your hands up in guard position"
            
            return None
        except KeyError:
            return None

    def _check_stance(self, pose_data: Dict) -> Optional[str]:
        """Check if stance is balanced and stable"""
        try:
            left_ankle = pose_data['left_ankle']
            right_ankle = pose_data['right_ankle']
            left_hip = pose_data['left_hip']
            right_hip = pose_data['right_hip']
            
            # Check foot positioning
            foot_distance = abs(left_ankle['x'] - right_ankle['x'])
            
            # Feet should be shoulder-width apart
            shoulder_width = abs(left_hip['x'] - right_hip['x'])
            
            if foot_distance < shoulder_width * 0.7:
                return "Widen your stance for better balance"
            elif foot_distance > shoulder_width * 1.5:
                return "Narrow your stance slightly"
            
            return None
        except KeyError:
            return None

    def _check_head_position(self, pose_data: Dict) -> Optional[str]:
        """Check head and chin position"""
        try:
            nose = pose_data['nose']
            left_shoulder = pose_data['left_shoulder']
            right_shoulder = pose_data['right_shoulder']
            
            # Head should be centered over shoulders
            shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            head_offset = abs(nose['x'] - shoulder_center_x)
            
            if head_offset > 0.1:  # 10% tolerance
                return "Keep your head centered over your shoulders"
            
            # Check if chin is tucked (nose should be above shoulder line)
            shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            if nose['y'] > shoulder_y:
                return "Tuck your chin down for protection"
            
            return None
        except KeyError:
            return None

    def _check_movement_patterns(self) -> Optional[str]:
        """Analyze movement patterns over time"""
        if len(self.pose_history) < 10:
            return None
        
        try:
            # Check for foot movement (good)
            recent_poses = self.pose_history[-10:]
            left_foot_positions = [pose['pose_data']['left_ankle']['x'] for pose in recent_poses]
            right_foot_positions = [pose['pose_data']['right_ankle']['x'] for pose in recent_poses]
            
            # Calculate movement variance
            left_variance = np.var(left_foot_positions)
            right_variance = np.var(right_foot_positions)
            
            # If very little movement, encourage footwork
            if left_variance < 0.001 and right_variance < 0.001:
                return "Add more footwork and movement"
            
            # Check for head movement
            head_positions = [pose['pose_data']['nose']['x'] for pose in recent_poses]
            head_variance = np.var(head_positions)
            
            if head_variance < 0.0005:
                return "Practice head movement and slipping"
            
            return None
        except (KeyError, IndexError):
            return None

    def generate_feedback(self, errors: List[str]) -> str:
        """
        Generate concise feedback message from current errors
        Prioritizes most important issues
        """
        if not errors:
            positive_feedback = [
                "Great form! Keep it up!",
                "Nice movement, stay focused!",
                "Good technique, maintain rhythm!",
                "Excellent guard position!",
                "Perfect stance, keep moving!"
            ]
            return np.random.choice(positive_feedback)
        
        # Prioritize errors by importance
        error_priority = {
            "Keep your hands up in guard position": 1,  # Most important
            "Tuck your chin down for protection": 2,
            "Keep your head centered over your shoulders": 3,
            "Widen your stance for better balance": 4,
            "Narrow your stance slightly": 4,
            "Add more footwork and movement": 5,
            "Practice head movement and slipping": 6
        }
        
        # Sort errors by priority
        prioritized_errors = sorted(errors, key=lambda x: error_priority.get(x, 99))
        
        # Return the highest priority error
        return prioritized_errors[0]

    def reset_analysis(self):
        """Reset analysis state for new session"""
        self.frame_count = 0
        self.pose_history = []
        self.current_errors = []
        self.feedback_history = []
        self.last_analysis_time = datetime.now()
        logger.info("Analysis state reset")

class StreamingGeminiClient:
    """Gemini client optimized for real-time streaming analysis"""
    
    def __init__(self):
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.conversation_history = []
        
        logger.info("✅ StreamingGeminiClient initialized")
    
    async def analyze_pose_stream(self, pose_data: Dict, errors: List[str]) -> str:
        """
        Analyze pose data in real-time and generate feedback
        Returns concise feedback suitable for TTS
        """
        try:
            # Create prompt for real-time analysis
            prompt = f"""
You are a live boxing coach giving real-time feedback. Be concise and actionable.

Current pose analysis shows these issues: {', '.join(errors) if errors else 'No major issues detected'}

Provide ONE specific instruction in 8 words or less. Examples:
- "Keep hands up near your face"
- "Tuck chin down for protection" 
- "Widen stance for better balance"
- "Add more head movement"
- "Great form, keep it up"

Response should be direct coaching advice only, no explanations.
"""

            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            feedback = response.text.strip()
            
            # Ensure feedback is concise (max 50 characters)
            if len(feedback) > 50:
                feedback = feedback[:47] + "..."
            
            return feedback
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            # Fallback to rule-based feedback
            if errors:
                return errors[0]
            return "Keep training, you're doing great!"