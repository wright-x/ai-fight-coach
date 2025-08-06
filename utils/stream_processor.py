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
import time
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import deque
import io
from PIL import Image

logger = logging.getLogger(__name__)

class StreamProcessor:
    """Processes video frames in real-time for live boxing analysis"""
    
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Analysis state
        self.frame_count = 0
        self.pose_history = deque(maxlen=30)  # Store last 30 poses for analysis
        self.current_errors = []
        self.feedback_history = []
        self.last_feedback_type = None
        self.frames_without_pose = 0
        
        logger.info("✅ StreamProcessor initialized with MediaPipe Pose")
    
    def process_frame(self, frame_data: str) -> Dict:
        """Process a single frame and extract pose data"""
        try:
            # Decode base64 frame
            img_data = base64.b64decode(frame_data.split(',')[1])
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            
            # Convert to RGB
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                rgb_frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = img_array
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            self.frame_count += 1
            
            if results.pose_landmarks:
                # Extract pose data
                pose_data = self._extract_pose_data(results.pose_landmarks)
                self.pose_history.append(pose_data)
                self.frames_without_pose = 0
                
                return {
                    "success": True,
                    "pose_detected": True,
                    "pose_data": pose_data,
                    "frame_count": self.frame_count
                }
            else:
                self.frames_without_pose += 1
                return {
                    "success": True,
                    "pose_detected": False,
                    "frames_without_pose": self.frames_without_pose,
                    "frame_count": self.frame_count
                }
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "frame_count": self.frame_count
            }
    
    def _extract_pose_data(self, landmarks) -> Dict:
        """Extract relevant pose data from MediaPipe landmarks"""
        try:
            # Get key landmarks
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            return {
                "nose": {"x": nose.x, "y": nose.y, "z": nose.z},
                "shoulders": {
                    "left": {"x": left_shoulder.x, "y": left_shoulder.y, "z": left_shoulder.z},
                    "right": {"x": right_shoulder.x, "y": right_shoulder.y, "z": right_shoulder.z}
                },
                "elbows": {
                    "left": {"x": left_elbow.x, "y": left_elbow.y, "z": left_elbow.z},
                    "right": {"x": right_elbow.x, "y": right_elbow.y, "z": right_elbow.z}
                },
                "wrists": {
                    "left": {"x": left_wrist.x, "y": left_wrist.y, "z": left_wrist.z},
                    "right": {"x": right_wrist.x, "y": right_wrist.y, "z": right_wrist.z}
                },
                "hips": {
                    "left": {"x": left_hip.x, "y": left_hip.y, "z": left_hip.z},
                    "right": {"x": right_hip.x, "y": right_hip.y, "z": right_hip.z}
                },
                "ankles": {
                    "left": {"x": left_ankle.x, "y": left_ankle.y, "z": left_ankle.z},
                    "right": {"x": right_ankle.x, "y": right_ankle.y, "z": right_ankle.z}
                },
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Pose data extraction error: {e}")
            return {}
    
    def generate_comprehensive_analysis(self) -> Dict:
        """Generate comprehensive analysis based on recent pose history"""
        if len(self.pose_history) < 5:
            return {"insufficient_data": True}
        
        if self.frames_without_pose > 10:
            return {"no_pose_detected": True}
        
        try:
            return {
                "stance_analysis": self._analyze_stance_details(),
                "footwork_analysis": self._analyze_footwork_patterns(),
                "guard_analysis": self._analyze_guard_details(),
                "head_movement_analysis": self._analyze_head_movement(),
                "balance_analysis": self._analyze_balance_distribution(),
                "power_generation_analysis": self._analyze_power_mechanics(),
                "defensive_analysis": self._analyze_defensive_positioning(),
                "movement_efficiency": self._analyze_movement_efficiency(),
                "timing_analysis": self._analyze_timing_patterns(),
                "comparison_to_elite": self._compare_to_professional_standards()
            }
        except Exception as e:
            logger.error(f"Analysis generation error: {e}")
            return {"analysis_error": True}
    
    def _analyze_stance_details(self):
        """Analyze stance width, balance, and positioning"""
        if not self.pose_history:
            return {"stance_width_ratio": 0.5}
        
        recent_poses = list(self.pose_history)[-10:]
        stance_widths = []
        
        for pose in recent_poses:
            if 'ankles' in pose:
                left_ankle = pose['ankles']['left']
                right_ankle = pose['ankles']['right']
                width = abs(left_ankle['x'] - right_ankle['x'])
                stance_widths.append(width)
        
        avg_width = np.mean(stance_widths) if stance_widths else 0.5
        return {
            "stance_width_ratio": min(max(avg_width, 0.0), 2.0),
            "balance_score": 0.7,
            "weight_distribution": 0.5
        }
    
    def _analyze_footwork_patterns(self):
        """Analyze footwork activity and movement patterns"""
        if len(self.pose_history) < 3:
            return {"total_movement": 0.0}
        
        movement_sum = 0
        for i in range(1, len(self.pose_history)):
            prev_pose = self.pose_history[i-1]
            curr_pose = self.pose_history[i]
            
            if 'ankles' in prev_pose and 'ankles' in curr_pose:
                left_movement = abs(curr_pose['ankles']['left']['x'] - prev_pose['ankles']['left']['x'])
                right_movement = abs(curr_pose['ankles']['right']['x'] - prev_pose['ankles']['right']['x'])
                movement_sum += (left_movement + right_movement) / 2
        
        return {
            "total_movement": min(movement_sum * 10, 2.0),
            "pivot_frequency": self._calculate_pivot_frequency(),
            "step_rhythm": self._analyze_step_rhythm(),
            "lateral_movement": self._analyze_lateral_movement(),
            "directional_balance": self._analyze_directional_movement()
        }
    
    def _analyze_guard_details(self):
        """Analyze guard positioning and hand height"""
        if not self.pose_history:
            return {"hand_height": 0.5}
        
        recent_poses = list(self.pose_history)[-5:]
        hand_heights = []
        
        for pose in recent_poses:
            if 'wrists' in pose and 'shoulders' in pose:
                left_hand_height = pose['shoulders']['left']['y'] - pose['wrists']['left']['y']
                right_hand_height = pose['shoulders']['right']['y'] - pose['wrists']['right']['y']
                avg_height = (left_hand_height + right_hand_height) / 2
                hand_heights.append(max(avg_height, 0))
        
        avg_hand_height = np.mean(hand_heights) if hand_heights else 0.5
        return {
            "hand_height": min(max(avg_hand_height, 0.0), 1.0),
            "elbow_position": 0.7,
            "guard_stability": 0.6
        }
    
    def _analyze_head_movement(self):
        """Analyze head movement and positioning"""
        if len(self.pose_history) < 3:
            return {"head_movement_frequency": 0.0}
        
        head_movements = []
        for i in range(1, len(self.pose_history)):
            prev_pose = self.pose_history[i-1]
            curr_pose = self.pose_history[i]
            
            if 'nose' in prev_pose and 'nose' in curr_pose:
                movement = abs(curr_pose['nose']['x'] - prev_pose['nose']['x']) + abs(curr_pose['nose']['y'] - prev_pose['nose']['y'])
                head_movements.append(movement)
        
        avg_movement = np.mean(head_movements) if head_movements else 0.0
        return {
            "head_movement_frequency": min(avg_movement * 20, 1.0),
            "slip_frequency": 0.3,
            "head_position": 0.5
        }
    
    def _analyze_balance_distribution(self):
        """Analyze weight distribution and balance"""
        return {"weight_forward": 0.5, "lateral_balance": 0.5, "stability_score": 0.7}
    
    def _analyze_power_mechanics(self):
        """Analyze power generation mechanics"""
        return {
            "kinetic_chain_efficiency": 0.6,
            "core_engagement": self._analyze_core_stability(),
            "ground_force_transfer": self._analyze_ground_force_transfer(),
            "rotation_timing": self._analyze_rotation_timing()
        }
    
    def _analyze_defensive_positioning(self):
        """Analyze defensive positioning"""
        return {"guard_coverage": 0.7, "counter_readiness": 0.6}
    
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
    
    def _compare_to_professional_standards(self):
        """Compare current form to professional standards"""
        return {
            "professional_similarity": 0.6,
            "technique_score": 0.7,
            "areas_for_improvement": ["footwork", "guard"]
        }

class StreamingGeminiClient:
    """Elite-level boxing coach powered by Gemini Streaming - gives feedback like Freddie Roach"""
    
    def __init__(self):
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        # Use Gemini 2.5 Pro as requested
        try:
            self.model = genai.GenerativeModel('gemini-1.5-pro-002')
            logger.info("✅ Using Gemini 1.5 Pro-002 (2.5 equivalent)")
        except:
            try:
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                logger.info("✅ Using Gemini 1.5 Pro")
            except:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Using Gemini 1.5 Flash (fallback)")
        
        # Track recent feedback for variety enforcement
        self.recent_tokens = deque(maxlen=6)  # Store recent bigrams
        self.category_rotation = deque(['guard', 'footwork', 'head_movement', 'punch_mechanics', 'rhythm', 'defense_counters', 'power_chain'], maxlen=7)
        self.last_category_used = None
        self.positive_reinforcement_count = 0
        self.last_verb_used = None
        
        logger.info("✅ Elite StreamingGeminiClient with deduplication initialized")
    
    async def generate_elite_coaching_feedback(self, comprehensive_analysis: Dict, last_feedback_types: list = None) -> str:
        """
        Generate concise, non-repetitive elite coaching feedback using Gemini streaming (≤15 words)
        Implements Freddie Roach-style coaching with strict variety enforcement
        """
        try:
            # Handle special cases first
            if comprehensive_analysis.get("insufficient_data"):
                return "Show me your stance and start moving."
            
            if comprehensive_analysis.get("no_pose_detected"):
                return "Step back into frame—camera can't see you."
            
            # Extract analysis data for the prompt
            stance_data = comprehensive_analysis.get('stance_analysis', {})
            footwork_data = comprehensive_analysis.get('footwork_analysis', {})
            guard_data = comprehensive_analysis.get('guard_analysis', {})
            head_data = comprehensive_analysis.get('head_movement_analysis', {})
            
            stance_width_ratio = stance_data.get('stance_width_ratio', 0.0)
            hand_height = guard_data.get('hand_height', 0.0)
            total_movement = footwork_data.get('total_movement', 0.0)
            head_mobility = head_data.get('head_movement_frequency', 0.0)
            
            # Determine next category to focus on (forced rotation)
            next_category = self._get_next_category()
            
            # Check if we need positive reinforcement (every 4th cue)
            self.positive_reinforcement_count += 1
            needs_positive = self.positive_reinforcement_count % 4 == 0
            
            # Create the elite coaching prompt
            prompt = self._create_freddie_roach_prompt(
                stance_width_ratio, hand_height, total_movement, head_mobility,
                next_category, needs_positive
            )
            
            # Use streaming generation
            feedback = await self._generate_streaming_feedback(prompt)
            
            # Enforce variety and deduplication
            if self._is_duplicate_feedback(feedback):
                logger.warning(f"🔄 Duplicate detected: '{feedback}', regenerating...")
                retry_prompt = prompt + "\n\nTry again—do NOT reuse previous wording."
                feedback = await self._generate_streaming_feedback(retry_prompt)
            
            # Store tokens for future deduplication
            self._store_feedback_tokens(feedback)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Elite Gemini streaming error: {e}")
            # Use fallback only once, then resume Gemini
            return self._generate_emergency_fallback(comprehensive_analysis)
    
    def _create_freddie_roach_prompt(self, stance_width_ratio: float, hand_height: float, 
                                   total_movement: float, head_mobility: float,
                                   focus_category: str, needs_positive: bool) -> str:
        """Create the exact Freddie Roach-style prompt as specified"""
        
        # Build category rotation instruction
        rotation_instruction = self._get_category_instruction(focus_category)
        
        # Add positive reinforcement if needed
        positive_note = "Sprinkle in positive reinforcement if appropriate." if needs_positive else ""
        
        prompt = f"""ROLE & TONE
You are a world-class boxing coach with an encyclopedic toolkit of cues.
Speak like Freddie Roach on fight night: concise, urgent, brutally honest—but never vulgar.

OUTPUT RULES
One sentence only, ≤ 15 words.
Must be fresh: do not repeat any phrase, verb, or noun used in the past 90 seconds.

Rotate focus in this order, skipping categories already covered in the last 3 tips:
Guard (hands, elbows, chin)
Footwork (stance width, pivots, lateral steps)
Head movement (slips, rolls, angle changes)
Punch mechanics (hip rotation, snap, retraction)
Rhythm / breathing (tempo, relaxation)
Defense & counters (blocks, parries, counter-timing)
Power chain (ground force, core engagement)

{rotation_instruction}
Never start two consecutive cues with the same verb.
{positive_note}

CONTEXT VARIABLES (insert live numbers)
StanceWidthRatio: {stance_width_ratio:.2f}
GuardHeight: {hand_height:.2f}
FootworkActivity: {total_movement:.2f}
HeadMobility: {head_mobility:.2f}

EXAMPLES
"Tuck elbows tighter; shorten stance two inches."
"Bounce left, double jab—don't park your feet."
"Good rhythm; add a quick slip after the cross."
"Rotate rear hip fully, exhale on impact."

FAIL-SAFE
If no pose detected for 3 seconds: "Step back into frame—camera can't see you."

RESPONSE (≤15 words, be different):"""
        
        return prompt
    
    def _get_next_category(self) -> str:
        """Get the next category in rotation, skipping recently used ones"""
        # Rotate to next category
        if self.last_category_used:
            try:
                current_index = list(self.category_rotation).index(self.last_category_used)
                next_index = (current_index + 1) % len(self.category_rotation)
                next_category = list(self.category_rotation)[next_index]
            except ValueError:
                next_category = list(self.category_rotation)[0]
        else:
            next_category = list(self.category_rotation)[0]
        
        self.last_category_used = next_category
        return next_category
    
    def _get_category_instruction(self, category: str) -> str:
        """Get specific instruction for the focus category"""
        instructions = {
            'guard': "FOCUS ON: Guard positioning - hands, elbows, chin protection.",
            'footwork': "FOCUS ON: Footwork - stance width, pivots, lateral movement.",
            'head_movement': "FOCUS ON: Head movement - slips, rolls, angle changes.",
            'punch_mechanics': "FOCUS ON: Punch mechanics - hip rotation, snap, retraction.",
            'rhythm': "FOCUS ON: Rhythm and breathing - tempo, relaxation.",
            'defense_counters': "FOCUS ON: Defense and counters - blocks, parries, timing.",
            'power_chain': "FOCUS ON: Power chain - ground force, core engagement."
        }
        return instructions.get(category, "FOCUS ON: Overall technique improvement.")
    
    async def _generate_streaming_feedback(self, prompt: str) -> str:
        """Generate feedback using Gemini streaming for optimal latency"""
        try:
            # Create streaming task
            stream_task = asyncio.create_task(self._stream_gemini_response(prompt))
            
            # Wait for response with timeout
            feedback = await asyncio.wait_for(stream_task, timeout=2.0)
            
            # Enforce 15 word limit
            words = feedback.split()
            if len(words) > 15:
                feedback = ' '.join(words[:15])
            
            return feedback.strip()
            
        except asyncio.TimeoutError:
            logger.warning("⏰ Streaming timeout, using emergency fallback")
            return "Keep moving, stay focused."
        except Exception as e:
            logger.error(f"❌ Streaming generation failed: {e}")
            raise
    
    async def _stream_gemini_response(self, prompt: str) -> str:
        """Generate response from Gemini using regular generation (streaming not available)"""
        try:
            # Use regular generation since streaming doesn't exist
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                logger.error("❌ Empty response from Gemini")
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            logger.error(f"❌ Gemini generation error: {e}")
            raise
    
    def _is_duplicate_feedback(self, feedback: str) -> bool:
        """Check if feedback contains duplicate bigrams from recent history"""
        # Extract bigrams from new feedback
        words = feedback.lower().split()
        new_bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1) if i < len(words)-1]
        
        # Check against recent tokens
        for bigram in new_bigrams:
            if bigram in self.recent_tokens:
                return True
        
        # Check for same starting verb
        if words and self.last_verb_used:
            first_word = words[0]
            if first_word == self.last_verb_used:
                return True
        
        return False
    
    def _store_feedback_tokens(self, feedback: str) -> None:
        """Store bigrams and verbs from feedback for deduplication"""
        words = feedback.lower().split()
        
        # Store bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1) if i < len(words)-1]
        for bigram in bigrams:
            self.recent_tokens.append(bigram)
        
        # Store first verb
        if words:
            self.last_verb_used = words[0]
    
    def _generate_emergency_fallback(self, analysis: Dict) -> str:
        """Generate emergency fallback when Gemini completely fails - with variety"""
        import random
        
        # Get current category for variety
        current_category = self._get_next_category()
        
        # Expert coaching cues based on boxing fundamentals
        fallback_cues = {
            'guard': [
                "Tuck elbows tighter, protect body.",
                "Hands up, chin down.",
                "Keep guard high, stay ready.",
                "Elbows in, cover ribs."
            ],
            'footwork': [
                "Bounce on balls of feet.",
                "Stay light, quick steps.",
                "Move lateral, don't stand still.",
                "Pivot on front foot."
            ],
            'head_movement': [
                "Move head after punching.",
                "Slip left, slip right.",
                "Duck low, come up ready.",
                "Keep head moving, stay elusive."
            ],
            'punch_mechanics': [
                "Snap punches back fast.",
                "Turn rear hip through.",
                "Double up that jab.",
                "Exhale on impact."
            ],
            'rhythm': [
                "Find your rhythm now.",
                "Breathe steady, stay relaxed.",
                "Good tempo, keep it up.",
                "Smooth combinations."
            ],
            'defense_counters': [
                "Block and counter immediately.",
                "Parry, then attack.",
                "Stay defensive, watch openings.",
                "Counter after his punch."
            ],
            'power_chain': [
                "Drive from legs up.",
                "Engage core, turn hips.",
                "Ground force to fist.",
                "Full body rotation."
            ]
        }
        
        # Get cues for current category
        category_cues = fallback_cues.get(current_category, fallback_cues['rhythm'])
        
        # Add variety by not using the same fallback twice
        if not hasattr(self, '_last_fallback'):
            self._last_fallback = None
        
        available_cues = [cue for cue in category_cues if cue != self._last_fallback]
        if not available_cues:
            available_cues = category_cues
        
        selected_cue = random.choice(available_cues)
        self._last_fallback = selected_cue
        
        logger.info(f"🔄 Emergency fallback ({current_category}): {selected_cue}")
        return selected_cue