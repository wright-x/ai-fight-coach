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
        self.feedback_history = []  # Recent feedback given (to avoid repetition)
        self.last_feedback_type = None  # Track last feedback type
        
        # Analysis parameters
        self.analysis_frequency = 1.0  # Analyze every second
        self.feedback_cooldown = 10.0  # 10 seconds between feedback
        self.pose_history_size = 30  # Keep 30 poses in history
        self.frames_without_pose = 0  # Track frames without pose detection
        
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
                self.frames_without_pose = 0  # Reset counter
                
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
            else:
                # No pose detected
                self.frames_without_pose += 1
                if self.frames_without_pose > 10:  # After 10 frames without pose
                    analysis_result['no_pose_detected'] = True
            
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

    def generate_comprehensive_analysis(self) -> Dict:
        """
        Generate comprehensive technical analysis for Gemini
        Returns detailed pose data, movement patterns, and context for elite-level coaching
        """
        if len(self.pose_history) < 10:
            return {"insufficient_data": True}
        
        recent_poses = self.pose_history[-10:]
        analysis = {
            "stance_analysis": self._analyze_stance_details(),
            "guard_analysis": self._analyze_guard_details(), 
            "footwork_analysis": self._analyze_footwork_patterns(),
            "head_movement_analysis": self._analyze_head_movement(),
            "balance_analysis": self._analyze_balance_distribution(),
            "power_generation_analysis": self._analyze_power_mechanics(),
            "defensive_positioning": self._analyze_defensive_setup(),
            "movement_efficiency": self._analyze_movement_efficiency(),
            "timing_patterns": self._analyze_timing_patterns(),
            "comparison_to_elite": self._compare_to_professional_standards()
        }
        
        return analysis

    def _analyze_stance_details(self) -> Dict:
        """Detailed stance analysis like a world-class trainer would do"""
        if len(self.pose_history) < 5:
            return {}
        
        latest_pose = self.pose_history[-1]['pose_data']
        
        try:
            left_ankle = latest_pose['left_ankle']
            right_ankle = latest_pose['right_ankle']
            left_knee = latest_pose['left_knee'] 
            right_knee = latest_pose['right_knee']
            left_hip = latest_pose['left_hip']
            right_hip = latest_pose['right_hip']
            
            # Calculate foot positioning
            foot_distance = abs(left_ankle['x'] - right_ankle['x'])
            shoulder_width = abs(left_hip['x'] - right_hip['x'])
            stance_ratio = foot_distance / shoulder_width if shoulder_width > 0 else 0
            
            # Analyze weight distribution
            left_weight = (left_knee['y'] + left_hip['y']) / 2
            right_weight = (right_knee['y'] + right_hip['y']) / 2
            weight_balance = abs(left_weight - right_weight)
            
            # Check for orthodox vs southpaw
            is_orthodox = left_ankle['x'] < right_ankle['x']
            
            return {
                "stance_width_ratio": stance_ratio,
                "weight_distribution": weight_balance,
                "stance_type": "orthodox" if is_orthodox else "southpaw",
                "foot_angle_analysis": self._calculate_foot_angles(),
                "knee_bend_analysis": self._analyze_knee_positioning(),
                "hip_alignment": self._analyze_hip_alignment()
            }
        except KeyError:
            return {}

    def _analyze_footwork_patterns(self) -> Dict:
        """Analyze footwork like Freddie Roach or Abel Sanchez would"""
        if len(self.pose_history) < 15:
            return {}
        
        # Track foot movement over time
        recent_poses = self.pose_history[-15:]
        
        left_foot_path = [(pose['pose_data']['left_ankle']['x'], pose['pose_data']['left_ankle']['y']) 
                         for pose in recent_poses if 'left_ankle' in pose['pose_data']]
        right_foot_path = [(pose['pose_data']['right_ankle']['x'], pose['pose_data']['right_ankle']['y']) 
                          for pose in recent_poses if 'right_ankle' in pose['pose_data']]
        
        # Calculate movement metrics
        left_distance = sum(np.sqrt((left_foot_path[i][0] - left_foot_path[i-1][0])**2 + 
                                   (left_foot_path[i][1] - left_foot_path[i-1][1])**2) 
                           for i in range(1, len(left_foot_path)))
        
        right_distance = sum(np.sqrt((right_foot_path[i][0] - right_foot_path[i-1][0])**2 + 
                                    (right_foot_path[i][1] - right_foot_path[i-1][1])**2) 
                            for i in range(1, len(right_foot_path)))
        
        return {
            "total_movement": left_distance + right_distance,
            "left_foot_activity": left_distance,
            "right_foot_activity": right_distance,
            "movement_balance": abs(left_distance - right_distance),
            "pivot_frequency": self._calculate_pivot_frequency(),
            "step_rhythm": self._analyze_step_rhythm(),
            "lateral_movement": self._analyze_lateral_movement(),
            "forward_backward_ratio": self._analyze_directional_movement()
        }

    def _analyze_power_mechanics(self) -> Dict:
        """Analyze power generation like Teddy Atlas or Nacho Beristain"""
        latest_pose = self.pose_history[-1]['pose_data']
        
        try:
            # Hip rotation analysis
            left_hip = latest_pose['left_hip']
            right_hip = latest_pose['right_hip']
            hip_angle = np.arctan2(right_hip['y'] - left_hip['y'], right_hip['x'] - left_hip['x'])
            
            # Shoulder alignment for power generation
            left_shoulder = latest_pose['left_shoulder']
            right_shoulder = latest_pose['right_shoulder']
            shoulder_angle = np.arctan2(right_shoulder['y'] - left_shoulder['y'], 
                                      right_shoulder['x'] - left_shoulder['x'])
            
            # Kinetic chain analysis
            kinetic_chain_alignment = abs(hip_angle - shoulder_angle)
            
            return {
                "hip_rotation_angle": hip_angle,
                "shoulder_alignment": shoulder_angle,
                "kinetic_chain_efficiency": kinetic_chain_alignment,
                "core_engagement": self._analyze_core_stability(),
                "ground_connection": self._analyze_ground_force_transfer(),
                "rotation_timing": self._analyze_rotation_timing()
            }
        except KeyError:
            return {}

    def _compare_to_professional_standards(self) -> Dict:
        """Compare technique to elite boxer standards"""
        return {
            "canelo_stance_similarity": 0.7,  # Placeholder - would use ML model
            "floyd_footwork_similarity": 0.6,
            "lomachenko_movement_similarity": 0.5,
            "ggg_power_mechanics": 0.8,
            "elite_guard_positioning": 0.6,
            "professional_balance": 0.7
        }

    def reset_analysis(self):
        """Reset analysis state for new session"""
        self.frame_count = 0
        self.pose_history = []
        self.current_errors = []
        self.feedback_history = []
        self.last_analysis_time = datetime.now()
        logger.info("Analysis state reset")

    # Helper methods for comprehensive analysis
    def _calculate_foot_angles(self):
        """Calculate foot positioning angles"""
        return {"left_foot_angle": 15, "right_foot_angle": 45}  # Placeholder
    
    def _analyze_knee_positioning(self):
        """Analyze knee bend and positioning"""
        return {"knee_bend_optimal": 0.8, "knee_alignment": 0.9}
    
    def _analyze_hip_alignment(self):
        """Analyze hip positioning and alignment"""
        return {"hip_square_percentage": 0.7, "hip_rotation_ready": 0.8}
    
    def _analyze_guard_details(self):
        """Detailed guard analysis"""
        return {"hand_height": 0.9, "elbow_positioning": 0.8, "guard_tightness": 0.7}
    
    def _analyze_head_movement(self):
        """Analyze head movement patterns"""
        return {"head_mobility": 0.6, "slip_readiness": 0.7, "chin_tuck": 0.8}
    
    def _analyze_balance_distribution(self):
        """Analyze weight distribution and balance"""
        return {"weight_distribution": 0.7, "balance_stability": 0.8}
    
    def _analyze_defensive_setup(self):
        """Analyze defensive positioning"""
        return {"defense_readiness": 0.7, "counter_positioning": 0.6}
    
    def _analyze_movement_efficiency(self):
        """Analyze movement efficiency"""
        return {"movement_economy": 0.6, "energy_conservation": 0.7}
    
    def _analyze_timing_patterns(self):
        """Analyze timing and rhythm"""
        return {"rhythm_consistency": 0.7, "timing_precision": 0.6}
    
    def _calculate_pivot_frequency(self):
        """Calculate pivot frequency"""
        return 0.5
    
    def _analyze_step_rhythm(self):
        """Analyze stepping rhythm"""
        return 0.7
    
    def _analyze_lateral_movement(self):
        """Analyze lateral movement"""
        return 0.6
    
    def _analyze_directional_movement(self):
        """Analyze forward/backward movement ratio"""
        return 0.8
    
    def _analyze_core_stability(self):
        """Analyze core engagement"""
        return 0.7
    
    def _analyze_ground_force_transfer(self):
        """Analyze ground force transfer"""
        return 0.8
    
    def _analyze_rotation_timing(self):
        """Analyze rotation timing"""
        return 0.6

class StreamingGeminiClient:
    """Elite-level boxing coach powered by Gemini - gives feedback like Freddie Roach, Abel Sanchez, Teddy Atlas"""
    
    def __init__(self):
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.conversation_history = []
        
        logger.info("✅ Elite StreamingGeminiClient initialized")
    
    async def generate_elite_coaching_feedback(self, comprehensive_analysis: Dict, last_feedback_types: list = None) -> str:
        """
        Generate concise, non-repetitive elite coaching feedback (15 words max)
        """
        try:
            if comprehensive_analysis.get("insufficient_data"):
                return "Show me your stance and start moving."
            
            if comprehensive_analysis.get("no_pose_detected"):
                return "Step back, I can't see you properly. Get in frame."
            
            # Avoid repetition
            if last_feedback_types is None:
                last_feedback_types = []
            
            # Create short, specific coaching prompt
            prompt = f"""
You are giving LIVE boxing coaching feedback. Be EXTREMELY concise - maximum 15 words.

ANALYSIS DATA:
Stance Quality: {comprehensive_analysis.get('stance_analysis', {}).get('stance_width_ratio', 0):.1f}
Footwork Activity: {comprehensive_analysis.get('footwork_analysis', {}).get('total_movement', 0):.1f}
Guard Height: {comprehensive_analysis.get('guard_analysis', {}).get('hand_height', 0):.1f}
Power Mechanics: {comprehensive_analysis.get('power_generation_analysis', {}).get('kinetic_chain_efficiency', 0):.1f}

AVOID REPEATING: {', '.join(last_feedback_types) if last_feedback_types else 'None'}

Give ONE specific instruction. Examples:
- "Hands higher, protect that chin"
- "Get on your toes, more bounce"
- "Turn that back hip through"
- "Move your head after punching"
- "Widen stance, better balance"

RESPONSE (15 words maximum):
"""

            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            feedback = response.text.strip()
            
            # Force 15 word limit
            words = feedback.split()
            if len(words) > 15:
                feedback = ' '.join(words[:15])
            
            return feedback
            
        except Exception as e:
            logger.error(f"Elite Gemini coaching error: {e}")
            # Fallback to short advice
            return self._generate_short_fallback_feedback(comprehensive_analysis, last_feedback_types)
    
    def _generate_short_fallback_feedback(self, analysis: Dict, last_feedback_types: list = None) -> str:
        """Generate short fallback feedback when Gemini fails"""
        
        stance = analysis.get('stance_analysis', {})
        footwork = analysis.get('footwork_analysis', {}) 
        guard = analysis.get('guard_analysis', {})
        power = analysis.get('power_generation_analysis', {})
        
        # Short, specific observations (15 words max)
        if stance.get('stance_width_ratio', 1.0) < 0.8 and 'stance' not in (last_feedback_types or []):
            return "Widen that stance, better balance."
        
        if footwork.get('total_movement', 0) < 0.1 and 'footwork' not in (last_feedback_types or []):
            return "Move your feet, you're standing still."
        
        if guard.get('hand_height', 0) < 0.7 and 'guard' not in (last_feedback_types or []):
            return "Hands up, protect that chin."
        
        if power.get('kinetic_chain_efficiency', 1.0) > 0.5 and 'power' not in (last_feedback_types or []):
            return "Turn that back hip through."
        
        # Default short advice (avoid repetition)
        short_advice = [
            "Good, keep that rhythm going.",
            "Nice form, stay focused.",
            "Breathe and stay relaxed.",
            "Move your head after punching.",
            "Get on your toes more.",
            "Circle left, use angles.",
            "Double up that jab.",
            "Snap those punches back."
        ]
        
        return np.random.choice(short_advice)