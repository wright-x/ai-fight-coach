"""
Streaming coach module – rebuilt for low-latency, varied live coaching.

Public API preserved:
- class StreamProcessor
- def get_gemini_client()
- Streaming client exposes:
  - generate_elite_coaching_feedback(analysis_dict)
  - generate_elite_coaching_feedback_with_stream(analysis_dict, send_delta, send_final)
"""

import os
import cv2
import io
import re
import time
import json
import base64
import queue
import asyncio
import logging
import random
import threading
import numpy as np
import mediapipe as mp
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import deque
from queue import Queue

try:
    import google.generativeai as genai
    # Typed safety enums (preferred)
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
        _TYPED_SAFETY = True
    except Exception:  # pragma: no cover
        HarmCategory = None
        HarmBlockThreshold = None
        _TYPED_SAFETY = False
except Exception as _ge:  # pragma: no cover
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    _TYPED_SAFETY = False


logger = logging.getLogger(__name__)


# =========================
# Pose processor (unchanged)
# =========================
class StreamProcessor:
    """Processes video frames in real-time for live boxing analysis"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.frame_count = 0
        self.pose_history: deque = deque(maxlen=30)
        self.frames_without_pose = 0
        logger.info("✅ StreamProcessor initialized with MediaPipe Pose")

    def process_frame(self, frame_data: str) -> Dict:
        try:
            img_data = base64.b64decode(frame_data.split(",")[1])
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            # Ensure RGB for MediaPipe
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                rgb_frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = img_array

            results = self.pose.process(rgb_frame)
            self.frame_count += 1

            if results.pose_landmarks:
                pose_data = self._extract_pose_data(results.pose_landmarks)
                self.pose_history.append(pose_data)
                self.frames_without_pose = 0
                return {"success": True, "pose_detected": True, "pose_data": pose_data, "frame_count": self.frame_count}
            else:
                self.frames_without_pose += 1
                return {"success": True, "pose_detected": False, "frames_without_pose": self.frames_without_pose, "frame_count": self.frame_count}
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {"success": False, "error": str(e), "frame_count": self.frame_count}

    def _extract_pose_data(self, landmarks) -> Dict:
        try:
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            l_sh = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_wr = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            r_wr = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            l_hp = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            r_hp = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            l_an = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            r_an = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            return {
                "nose": {"x": nose.x, "y": nose.y, "z": nose.z},
                "shoulders": {
                    "left": {"x": l_sh.x, "y": l_sh.y, "z": l_sh.z},
                    "right": {"x": r_sh.x, "y": r_sh.y, "z": r_sh.z},
                },
                "wrists": {
                    "left": {"x": l_wr.x, "y": l_wr.y, "z": l_wr.z},
                    "right": {"x": r_wr.x, "y": r_wr.y, "z": r_wr.z},
                },
                "hips": {
                    "left": {"x": l_hp.x, "y": l_hp.y, "z": l_hp.z},
                    "right": {"x": r_hp.x, "y": r_hp.y, "z": r_hp.z},
                },
                "ankles": {
                    "left": {"x": l_an.x, "y": l_an.y, "z": l_an.z},
                    "right": {"x": r_an.x, "y": r_an.y, "z": r_an.z},
                },
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Pose data extraction error: {e}")
            return {}

    # Minimal analysis used by the coach
    def generate_comprehensive_analysis(self) -> Dict:
        if len(self.pose_history) < 5:
            return {"insufficient_data": True}
        if self.frames_without_pose > 10:
            return {"no_pose_detected": True}

        recent = list(self.pose_history)[-10:]
        # stance width proxy
        widths = []
        movements = []
        hand_heights = []
        head_moves = []
        for i, pose in enumerate(recent):
            if "ankles" in pose:
                l, r = pose["ankles"]["left"], pose["ankles"]["right"]
                widths.append(abs(l["x"] - r["x"]))
            if "wrists" in pose and "shoulders" in pose:
                lh = pose["shoulders"]["left"]["y"] - pose["wrists"]["left"]["y"]
                rh = pose["shoulders"]["right"]["y"] - pose["wrists"]["right"]["y"]
                hand_heights.append(max((lh + rh) / 2, 0))
            if i > 0 and "nose" in pose and "nose" in recent[i - 1]:
                prev = recent[i - 1]
                head_moves.append(abs(pose["nose"]["x"] - prev["nose"]["x"]) + abs(pose["nose"]["y"] - prev["nose"]["y"]))
            if i > 0 and "ankles" in pose and "ankles" in recent[i - 1]:
                prev = recent[i - 1]
                lm = abs(pose["ankles"]["left"]["x"] - prev["ankles"]["left"]["x"]) + abs(pose["ankles"]["left"]["y"] - prev["ankles"]["left"]["y"])
                rm = abs(pose["ankles"]["right"]["x"] - prev["ankles"]["right"]["x"]) + abs(pose["ankles"]["right"]["y"] - prev["ankles"]["right"]["y"])
                movements.append((lm + rm) / 2)

        stance_width_ratio = float(np.mean(widths)) if widths else 0.55
        hand_height = float(np.mean(hand_heights)) if hand_heights else 0.5
        total_movement = min(float(np.sum(movements)) * 5.0, 2.0)
        head_mobility = min(float(np.mean(head_moves)) * 20.0 if head_moves else 0.0, 1.0)

        return {
            "stance_analysis": {"stance_width_ratio": min(max(stance_width_ratio, 0.0), 2.0)},
            "guard_analysis": {"hand_height": min(max(hand_height, 0.0), 1.0)},
            "footwork_analysis": {"total_movement": total_movement},
            "head_movement_analysis": {"head_movement_frequency": head_mobility},
        }


# =============================
# Gemini streaming coach client
# =============================
_COACH_SINGLETON = None


def get_gemini_client():
    global _COACH_SINGLETON
    if _COACH_SINGLETON is None:
        _COACH_SINGLETON = _StreamingCoach()
        logger.info("🔥 Created singleton StreamingGeminiClient")
    return _COACH_SINGLETON


class _StreamingCoach:
    """Low-latency, varied live coaching with true token streaming and safe fallbacks."""

    BANNED = {"kid", "bro", "champ", "buddy", "pal", "dude", "man"}
    FALLBACKS = deque([
        "Shorten stance slightly; tuck elbows.",
        "Angle left, double jab—don’t square up.",
        "Tuck chin; bring rear hand to cheek.",
        "Small pivot, reset balance before punching.",
        "Roll under, step off the line.",
    ])

    def __init__(self) -> None:
        if genai is None:
            raise RuntimeError("google.generativeai is required")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY is required")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL_COACH", "gemini-2.5-flash")
        self.model = genai.GenerativeModel(model_name)
        self.stream_mode = os.getenv("GEMINI_STREAM_MODE", "final").lower()
        logger.info(f"🔥 Using {model_name} for live coaching (stream_mode: {self.stream_mode})")

        # Variety engine state
        self.last_5_sentences: deque[str] = deque(maxlen=5)
        self.verb_history: deque[str] = deque(maxlen=3)
        self.category_history: deque[str] = deque(maxlen=3)
        self.bigram_window: deque[Tuple[str, float]] = deque(maxlen=200)  # (bigram, ts)
        self.category_ring: deque[str] = deque([
            "guard", "footwork", "head_movement", "punch_mechanics", "rhythm", "defense", "power_chain"
        ])
        self.bigram_ttl_sec = 90.0

        # Metric rolling windows for praise check
        self.metric_history = {
            "hand_height": deque(maxlen=30),
            "stance_width_ratio": deque(maxlen=30),
            "total_movement": deque(maxlen=30),
            "head_mobility": deque(maxlen=30),
        }

        # Rate limit/backoff
        self.last_request_time = 0.0
        self.backoff_until = 0.0
        self.rate_limit_delay = 0.0  # active streaming path should not throttle

    # ---------------
    # Safety settings
    # ---------------
    def _safety_settings(self):
        if _TYPED_SAFETY and HarmCategory is not None:
            return [
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM},
            ]
        # Fallback string form
        return [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM"},
        ]

    def _gen_config(self):
        return {
            "temperature": 0.65,
            "top_p": 0.9,
            "max_output_tokens": 32,
            "response_mime_type": "text/plain",
        }

    # ------------------
    # Variety evaluation
    # ------------------
    def _first_word(self, text: str) -> str:
        words = text.strip().split()
        return words[0].lower() if words else ""

    def _lev_ratio(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j - 1] if a[i - 1] == b[j - 1] else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        dist = dp[m][n]
        return 1.0 - (dist / max(m, n))

    def _bigrams(self, text: str) -> List[str]:
        words = text.strip().lower().split()
        return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    def _update_bigrams(self, text: str) -> None:
        now = time.time()
        for b in self._bigrams(text):
            self.bigram_window.append((b, now))
        # TTL prune
        while self.bigram_window and (now - self.bigram_window[0][1]) > self.bigram_ttl_sec:
            self.bigram_window.popleft()

    def _violates_variety(self, candidate: str) -> bool:
        # Levenshtein against last sentences
        for prev in self.last_5_sentences:
            if self._lev_ratio(candidate.lower(), prev.lower()) > 0.75:
                return True
        # opening verb
        if self.verb_history and self._first_word(candidate) == self.verb_history[-1]:
            return True
        # bigram repeats
        recent = {b for (b, _) in self.bigram_window}
        for b in self._bigrams(candidate):
            if b in recent:
                return True
        return False

    def _rotate_category(self) -> str:
        # skip any used in last 3 tips
        for _ in range(len(self.category_ring)):
            cat = self.category_ring[0]
            self.category_ring.rotate(-1)
            if cat not in self.category_history:
                return cat
        return self.category_ring[0]

    # --------------
    # Prompt builder
    # --------------
    def _build_prompt(self, metrics: Dict, should_praise: bool) -> str:
        last_opening = self.verb_history[-1] if self.verb_history else "None"
        excluded = list(self.category_history)
        jitter = random.randint(1000, 9999)
        context_line = "This is non-violent sports training feedback for solo shadowboxing; fitness/technique only."
        return f"""{context_line}
ROLE
You are an elite boxing coach delivering live micro-cues. Be urgent, specific, and professional.

HARD BANS
Never use address terms or nicknames: kid, bro, champ, buddy, pal, dude, man.
No profanity. No emojis/hashtags or filler ("come on", "let's go", "you got this").

OUTPUT FORMAT
Return ONE sentence, plain text only, ≤ 15 words. Start with a direct, actionable cue.

VARIETY & REPETITION
Do NOT reuse the same opening word as the previous tip: {last_opening}.
Avoid any bigram used in the last 90 seconds (the app enforces; try your best).
If you cannot be fresh, switch focus area or rephrase.

FOCUS ROTATION (skip any in ExcludedCategories)
1) Guard (hands, elbows, chin)
2) Footwork (stance width, pivots, lateral movement)
3) Head movement (slips, rolls, angles)
4) Punch mechanics (hip rotation, snap, retraction)
5) Rhythm/breathing (tempo, relaxation)
6) Defense & counters (blocks, parries, timing)
7) Power chain (ground force, core engagement)

POSITIVE REINFORCEMENT (subtle, optional)
If metrics exceed target or improve ≥10% vs. 30-frame average, prefix with 1–2 words of praise.
Use praise at most 1 in 4 tips.

LIVE CONTEXT
StanceWidthRatio: {metrics['stance_width_ratio']:.2f} (target 0.55–0.70)
GuardHeight: {metrics['hand_height']:.2f}
FootworkActivity: {metrics['total_movement']:.2f}
HeadMobility: {metrics['head_mobility']:.2f}
ExcludedCategories: {excluded}
ShouldPraise: {should_praise}

FAIL-SAFES (emit exactly when true)
- If no pose >3s: Step back into frame—camera can't see you.
- If insufficient data (<5 frames): Show me your stance and start moving.

TONE EXAMPLES (style only; do not reuse wording)
"Tuck elbows; bring rear hand to cheek."
"Angle off right, double jab—don't square up."
"Good rhythm; add a slip after the cross."
"Rotate hip through, snap and retract fast."

RESPONSE
Output only the final one-sentence cue (≤15 words), plain text, nothing else.
PROMPT_JITTER_{jitter}"""

    # --------------
    # Sanitizer/token
    # --------------
    def _sanitize(self, text: str) -> Optional[str]:
        words = text.split()
        out: List[str] = []
        for w in words:
            c = w.lower().strip(",.!?;:\"'()[]{}")
            if c in self.BANNED:
                continue
            out.append(w)
        s = " ".join(out).strip()
        if not s or len(s.split()) < 2:
            return None
        return s

    def _safe_tokenize(self, chunk_text: str) -> List[str]:
        safe = []
        for raw in chunk_text.split():
            c = raw.lower().strip(",.!?;:\"'()[]{}")
            if c in self.BANNED:
                continue
            safe.append(raw)
        return safe

    # ------------------
    # Response extractors
    # ------------------
    def _extract_text_from_resp(self, resp) -> str:
        try:
            # Prefer resp.text if available
            t = getattr(resp, "text", None)
            if t:
                return t.strip()
            parts_out = []
            for c in getattr(resp, "candidates", []) or []:
                content = getattr(c, "content", None)
                if not content:
                    continue
                for p in getattr(content, "parts", []) or []:
                    tx = getattr(p, "text", None)
                    if tx:
                        parts_out.append(tx)
            return " ".join(parts_out).strip()
        except Exception:
            logger.debug("_extract_text_from_resp failed", exc_info=True)
            return ""

    def _extract_text_from_chunk(self, chunk) -> str:
        try:
            parts_out = []
            for c in getattr(chunk, "candidates", []) or []:
                content = getattr(c, "content", None)
                if not content:
                    continue
                for p in getattr(content, "parts", []) or []:
                    tx = getattr(p, "text", None)
                    if tx:
                        parts_out.append(tx)
            if parts_out:
                return " ".join(parts_out)
            t = getattr(chunk, "text", None)
            return t or ""
        except Exception:
            logger.debug("_extract_text_from_chunk failed", exc_info=True)
            return ""

    # ------------------------
    # Praise gating & metrics
    # ------------------------
    def _should_praise(self, metrics: Dict) -> bool:
        thresholds = {"hand_height": 0.65, "stance_width_ratio": 0.55, "total_movement": 0.30, "head_mobility": 0.20}
        if any(metrics.get(k, 0.0) > v for k, v in thresholds.items()):
            return random.random() < 0.25
        improving = False
        for k, v in metrics.items():
            hist = self.metric_history.get(k)
            if hist and len(hist) >= 10:
                avg = sum(hist) / len(hist)
                if avg and v > avg * 1.10:
                    improving = True
                    break
        return improving and (random.random() < 0.25)

    def _update_metric_history(self, metrics: Dict) -> None:
        for k, v in metrics.items():
            if k in self.metric_history:
                self.metric_history[k].append(v)

    # -----------------
    # Public entrypoints
    # -----------------
    async def generate_elite_coaching_feedback(self, analysis: Dict, last_feedback_types: List[str] | None = None) -> str:
        # Fail-safes
        if analysis.get("insufficient_data"):
            return "Show me your stance and start moving."
        if analysis.get("no_pose_detected"):
            return "Step back into frame—camera can't see you."

        # Metrics
        stance = analysis.get("stance_analysis", {})
        guard = analysis.get("guard_analysis", {})
        footw = analysis.get("footwork_analysis", {})
        head = analysis.get("head_movement_analysis", {})
        metrics = {
            "stance_width_ratio": float(stance.get("stance_width_ratio", 0.0)),
            "hand_height": float(guard.get("hand_height", 0.0)),
            "total_movement": float(footw.get("total_movement", 0.0)),
            "head_mobility": float(head.get("head_movement_frequency", 0.0)),
        }
        self._update_metric_history(metrics)
        should_praise = self._should_praise(metrics)

        prompt = self._build_prompt(metrics, should_praise)

        def _once():
            resp = self.model.generate_content(prompt, generation_config=self._gen_config(), safety_settings=self._safety_settings())
            txt = self._extract_text_from_resp(resp)
            if not txt:
                # debug once
                try:
                    cand0 = (getattr(resp, "candidates", None) or [None])[0]
                    finish_reason = getattr(cand0, "finish_reason", None)
                    safety = getattr(cand0, "safety_ratings", None)
                    prompt_fb = getattr(resp, "prompt_feedback", None)
                    parts_types = []
                    if cand0 and getattr(cand0, "content", None):
                        parts_types = [type(p).__name__ for p in getattr(cand0.content, "parts", []) or []]
                    logger.debug(f"[GEMINI DEBUG] finish_reason={finish_reason} parts={parts_types} safety={safety} prompt_feedback={prompt_fb}")
                except Exception:
                    logger.debug("[GEMINI DEBUG] inspect failed", exc_info=True)
            return txt

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _once)
        if not text:
            fb = self._rotating_fallback()
            return fb

        text = self._postprocess_sentence(text)
        if not text or self._violates_variety(text):
            # single retry with appended instruction
            text2 = await loop.run_in_executor(None, lambda: self._extract_text_from_resp(self.model.generate_content(prompt + "\nRephrase with a different opening verb and wording.", generation_config=self._gen_config(), safety_settings=self._safety_settings())))
            text = self._postprocess_sentence(text2) or self._rotating_fallback()

        self._record_sentence(text)
        return text

    async def generate_elite_coaching_feedback_with_stream(self, analysis: Dict, send_delta, send_final) -> str:
        # Fail-safes
        if analysis.get("insufficient_data"):
            await send_final("Show me your stance and start moving.")
            return "Show me your stance and start moving."
        if analysis.get("no_pose_detected"):
            await send_final("Step back into frame—camera can't see you.")
            return "Step back into frame—camera can't see you."

        # Metrics
        stance = analysis.get("stance_analysis", {})
        guard = analysis.get("guard_analysis", {})
        footw = analysis.get("footwork_analysis", {})
        head = analysis.get("head_movement_analysis", {})
        metrics = {
            "stance_width_ratio": float(stance.get("stance_width_ratio", 0.0)),
            "hand_height": float(guard.get("hand_height", 0.0)),
            "total_movement": float(footw.get("total_movement", 0.0)),
            "head_mobility": float(head.get("head_movement_frequency", 0.0)),
        }
        self._update_metric_history(metrics)
        should_praise = self._should_praise(metrics)
        prompt = self._build_prompt(metrics, should_praise)

        # Token streaming via worker thread + queue bridge
        q: Queue = Queue(maxsize=64)
        stop_evt = threading.Event()

        def _producer():
            try:
                start = time.time()
                stream = self.model.generate_content(prompt, generation_config=self._gen_config(), stream=True, safety_settings=self._safety_settings())
                first_sent = False
                for chunk in stream:
                    if stop_evt.is_set():
                        break
                    text = self._extract_text_from_chunk(chunk)
                    if not text:
                        continue
                    if not first_sent:
                        first_sent = True
                        logger.info(f"⏱️ first-token-latency={(time.time()-start)*1000:.0f}ms")
                    q.put(text)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)  # sentinel

        threading.Thread(target=_producer, daemon=True).start()

        assembled: List[str] = []
        loop = asyncio.get_running_loop()
        tokens_emitted = 0
        t0 = time.time()

        # Zero-chunk timeout -> one non-stream retry
        got_any = False
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                logger.warning(f"token stream error: {item}")
                break
            got_any = True
            tokens = self._safe_tokenize(str(item))
            if not tokens:
                continue
            tokens_emitted += len(tokens)
            delta = " ".join(tokens)
            assembled.extend(tokens)
            await send_delta(delta)

            # Prevent overrun – stop if more than ~20 words assembled
            if len(assembled) >= 20:
                break

        if not got_any and (time.time() - t0) >= 0.8:
            # metadata could be sent here if supported; skip to keep API stable
            logger.warning("⚠️ Zero token chunks within 800ms; retrying non-stream once")
            text = await self.generate_elite_coaching_feedback(analysis)
            await send_final(text)
            return text

        final_text = " ".join(assembled).strip()
        if not final_text:
            text = await self.generate_elite_coaching_feedback(analysis)
            await send_final(text)
            return text

        final_text = self._postprocess_sentence(final_text) or self._rotating_fallback()
        self._record_sentence(final_text)
        logger.info(f"✅ stream tokens={tokens_emitted} sentence='{final_text}'")
        await send_final(final_text)
        return final_text

    # --------------
    # Helpers
    # --------------
    def _postprocess_sentence(self, text: str) -> Optional[str]:
        s = re.sub(r"\s+", " ", text or "").strip()
        # enforce ≤ 15 words
        words = s.split()
        if len(words) > 15:
            s = " ".join(words[:15])
        s = self._sanitize(s) or ""
        if not s:
            return None
        return s

    def _record_sentence(self, text: str) -> None:
        self.last_5_sentences.append(text)
        self.verb_history.append(self._first_word(text))
        self._update_bigrams(text)
        # category guess for rotation memory
        lower = text.lower()
        if any(k in lower for k in ["chin", "guard", "elbow", "hand"]):
            self.category_history.append("guard")
        elif any(k in lower for k in ["foot", "step", "move", "pivot"]):
            self.category_history.append("footwork")
        elif any(k in lower for k in ["head", "slip", "roll"]):
            self.category_history.append("head_movement")
        elif any(k in lower for k in ["jab", "cross", "hook", "hip", "snap", "retract"]):
            self.category_history.append("punch_mechanics")
        elif any(k in lower for k in ["rhythm", "breathe", "tempo"]):
            self.category_history.append("rhythm")
        elif any(k in lower for k in ["block", "parry", "counter"]):
            self.category_history.append("defense")
        elif any(k in lower for k in ["power", "core", "ground"]):
            self.category_history.append("power_chain")

    def _rotating_fallback(self) -> str:
        self.FALLBACKS.rotate(-1)
        return self.FALLBACKS[0]

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
import random
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import deque
from queue import Queue, Empty
import io
from PIL import Image

# Sentinel for queue communication
_SENTINEL = object()

logger = logging.getLogger(__name__)

# Module-level singleton for StreamingGeminiClient
_gemini_client_instance = None

def get_gemini_client():
    """Get singleton instance of StreamingGeminiClient"""
    global _gemini_client_instance
    if _gemini_client_instance is None:
        _gemini_client_instance = StreamingGeminiClient()
        logger.info("🔥 Created singleton StreamingGeminiClient")
    return _gemini_client_instance

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
        # Use gemini-2.5-flash for high RPM micro-cues
        model_name = os.getenv("GEMINI_MODEL_COACH", "gemini-2.5-flash")
        self.model = genai.GenerativeModel(model_name)
        
        # Stream mode configuration
        self.stream_mode = os.getenv("GEMINI_STREAM_MODE", "final").lower()  # "final" or "tokens"
        
        logger.info(f"🔥 Using {model_name} for live coaching (stream_mode: {self.stream_mode})")
        
        # Hard variety guard - track last 5 full sentences
        self.last_5_responses = deque(maxlen=5)
        self.category_history = deque(maxlen=3)
        self.verb_history = deque(maxlen=3)
        self.retry_count = 0
        
        # Rate limiting and banned terms
        self.last_request_time = 0
        self.rate_limit_delay = 1.2  # seconds between requests
        self.backoff_until = 0
        self.banned_terms = {"kid", "bro", "champ", "buddy", "pal"}
        self.BANNED = {"kid", "bro", "champ", "buddy", "pal"}  # For sanitizer
        
        # Rolling averages for compliment logic
        self.metric_history = {
            'hand_height': deque(maxlen=30),
            'stance_width_ratio': deque(maxlen=30),
            'total_movement': deque(maxlen=30),
            'head_mobility': deque(maxlen=30)
        }
        self.positive_reinforcement_count = 0
        self.last_verb_used = None
        
        logger.info("✅ Elite StreamingGeminiClient with deduplication initialized")
    
    def _levenshtein_ratio(self, a: str, b: str) -> float:
        """Calculate Levenshtein similarity ratio between two strings"""
        if not a or not b:
            return 0.0
        
        # Simple Levenshtein implementation
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    def _get_first_verb(self, text: str) -> str:
        """Extract the first verb from feedback text"""
        words = text.strip().lower().split()
        return words[0] if words else ""
    
    def _sanitize_response(self, response: str) -> str:
        """Remove banned terms and clean up response - returns None if too short after sanitization"""
        # Remove banned terms (case-insensitive)
        words = response.split()
        filtered_words = []
        for word in words:
            # Strip punctuation for comparison
            clean_word = word.lower().strip(",.!?")
            if clean_word not in self.BANNED:
                filtered_words.append(word)
        
        # Reconstruct response
        out = " ".join(filtered_words)
        
        # Safety: if we removed too much, return None to trigger retry
        if not out or len(out.split()) < 2:
            return None
            
        return out.strip()
    
    def _should_praise(self, current_metrics: dict) -> bool:
        """Determine if we should add praise based on metrics"""
        if not any(self.metric_history.values()):
            return False
        
        # Check if any metric is above threshold or improving
        thresholds = {
            'hand_height': 0.7,  # Good guard height
            'stance_width_ratio': 0.6,  # Good stance width
            'total_movement': 0.3,  # Active footwork
            'head_mobility': 0.2   # Head movement
        }
        
        for metric, value in current_metrics.items():
            if metric in thresholds and value > thresholds[metric]:
                return random.random() < 0.25  # 25% chance when good
        
        # Check for improvement vs rolling average
        for metric, value in current_metrics.items():
            if metric in self.metric_history and len(self.metric_history[metric]) >= 10:
                avg = sum(self.metric_history[metric]) / len(self.metric_history[metric])
                if value > avg * 1.1:  # 10% improvement
                    return random.random() < 0.25
        
        return False
    
    def _check_variety(self, response_text: str) -> bool:
        """Check if response is too similar to recent ones"""
        # Check Levenshtein ratio
        for prev_response in self.last_5_responses:
            ratio = self._levenshtein_ratio(response_text.lower(), prev_response.lower())
            if ratio > 0.75:
                logger.warning(f"🔄 High similarity ratio: {ratio:.2f}")
                return False
        
        # Check first two words match
        current_words = response_text.strip().lower().split()[:2]
        for prev_response in self.last_5_responses:
            prev_words = prev_response.strip().lower().split()[:2]
            if len(current_words) >= 2 and len(prev_words) >= 2:
                if current_words == prev_words:
                    logger.warning(f"🔄 First two words match: {current_words}")
                    return False
        
        # Check verb repetition
        current_verb = self._get_first_verb(response_text)
        if current_verb in self.verb_history:
            logger.warning(f"🔄 Verb '{current_verb}' recently used")
            return False
            
        return True
    
    async def generate_elite_coaching_feedback(self, comprehensive_analysis: Dict, last_feedback_types: list = None) -> str:
        """Generate elite coaching feedback using gemini-2.5-flash with rate limiting and sanitization"""
        import time
        
        # Handle special cases first
        if comprehensive_analysis.get("insufficient_data"):
            return "Show me your stance and start moving."
        
        if comprehensive_analysis.get("no_pose_detected"):
            return "Step back into frame—camera can't see you."
        
        # Rate limiting check
        current_time = time.time()
        if current_time < self.backoff_until:
            logger.info(f"⏰ Rate limit backoff active until {self.backoff_until:.1f}s")
            return "Shadowbox four beats while I reconnect."
        
        if current_time - self.last_request_time < self.rate_limit_delay:
            logger.info(f"⏰ Rate limit delay: {self.rate_limit_delay - (current_time - self.last_request_time):.1f}s remaining")
            return "Keep moving, stay focused."
        
        # Update metric history
        stance_data = comprehensive_analysis.get('stance_analysis', {})
        guard_data = comprehensive_analysis.get('guard_analysis', {})
        footwork_data = comprehensive_analysis.get('footwork_analysis', {})
        head_data = comprehensive_analysis.get('head_movement_analysis', {})
        
        current_metrics = {
            'hand_height': guard_data.get('hand_height', 0.0),
            'stance_width_ratio': stance_data.get('stance_width_ratio', 0.0),
            'total_movement': footwork_data.get('total_movement', 0.0),
            'head_mobility': head_data.get('head_movement_frequency', 0.0)
        }
        
        for metric, value in current_metrics.items():
            self.metric_history[metric].append(value)
        
        # Determine if we should praise
        should_praise = self._should_praise(current_metrics)
        
        # Build the new coaching prompt
        excluded_categories = list(self.category_history)
        jitter = random.randint(1000, 9999)
        
        # Get last opening verb for variety enforcement
        last_opening_verb = self.verb_history[-1] if self.verb_history else "None"
        
        prompt = f"""ROLE
You are an elite boxing coach delivering live micro-cues. Be urgent, specific, and professional.

HARD BANS
Never use address terms or nicknames: kid, bro, champ, buddy, pal, dude, man.
No profanity. No hashtags, emojis, or filler ("come on", "let's go", "you got this").

OUTPUT FORMAT
Return ONE sentence, plain text only (no quotes, no colons, no code blocks), ≤ 15 words.
Start with a direct, actionable cue (imperative or concise declarative). Avoid hedging.

VARIETY & REPETITION
Do NOT reuse the same opening word as the previous tip: {last_opening_verb}.
Avoid any bigram that appeared in the last 90 seconds (the app enforces; you must try).
If you cannot be fresh, pick a different focus area (see rotation) or rephrase.

FOCUS ROTATION (skip any in ExcludedCategories)
1) Guard (hands, elbows, chin)
2) Footwork (stance width, pivots, lateral movement)
3) Head movement (slips, rolls, angles)
4) Punch mechanics (hip rotation, snap, retraction)
5) Rhythm/breathing (tempo, relaxation)
6) Defense & counters (blocks, parries, timing)
7) Power chain (ground force, core engagement)

POSITIVE REINFORCEMENT (subtle, optional)
If a metric is above target or improving ≥10% vs. 30-frame average, prefix with 1–2 words of praise
(e.g., "Nice guard." "Good rhythm.") then the cue. Use praise at most 1 in 4 tips.

LIVE CONTEXT (use these numbers to choose and phrase the cue)
StanceWidthRatio: {current_metrics['stance_width_ratio']:.2f}   (target 0.55–0.70)
GuardHeight:     {current_metrics['hand_height']:.2f}         (higher is better)
FootworkActivity:{current_metrics['total_movement']:.2f}
HeadMobility:    {current_metrics['head_mobility']:.2f}
ExcludedCategories: {excluded_categories}
ShouldPraise: {should_praise}

FAIL-SAFES (only emit these exact lines when true)
- If no pose detected for >3s: Step back into frame—camera can't see you.
- If insufficient data (<5 frames): Show me your stance and start moving.

TONE EXAMPLES (style only; do not reuse wording)
"Tuck elbows; bring rear hand to cheek."
"Angle off right, double jab—don't square up."
"Good rhythm; add a slip after the cross."
"Rotate hip through, snap and retract fast."

RESPONSE
Output only the final one-sentence cue (≤15 words), plain text, nothing else.
PROMPT_JITTER_{jitter}"""
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Update request time
                self.last_request_time = current_time
                
                # Generate response
                response_text = await self._generate_streaming_feedback(prompt)
                
                # Sanitize response
                sanitized_text = self._sanitize_response(response_text)
                
                # Check if sanitization removed too much
                if sanitized_text is None:
                    if attempt < max_retries:
                        logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} - response too short after sanitization")
                        prompt += "\nAvoid nicknames."
                        continue
                    else:
                        return "Adjust distance, stay active."
                
                response_text = sanitized_text
                
                # Enforce 15 word limit
                words = response_text.split()
                if len(words) > 15:
                    response_text = " ".join(words[:15])
                
                # Check variety
                if self._check_variety(response_text):
                    # Add to history
                    self.last_5_responses.append(response_text)
                    verb = self._get_first_verb(response_text)
                    if verb:
                        self.verb_history.append(verb)
                    
                    # Add category to history
                    if any(word in response_text.lower() for word in ['chin', 'guard', 'hands', 'elbow']):
                        self.category_history.append('guard')
                    elif any(word in response_text.lower() for word in ['foot', 'step', 'move', 'bounce']):
                        self.category_history.append('footwork')
                    elif any(word in response_text.lower() for word in ['head', 'slip', 'duck']):
                        self.category_history.append('head_movement')
                    elif any(word in response_text.lower() for word in ['punch', 'jab', 'cross', 'hook']):
                        self.category_history.append('punch_mechanics')
                    elif any(word in response_text.lower() for word in ['rhythm', 'breathe', 'tempo']):
                        self.category_history.append('rhythm')
                    elif any(word in response_text.lower() for word in ['block', 'parry', 'counter']):
                        self.category_history.append('defense')
                    elif any(word in response_text.lower() for word in ['power', 'hip', 'core']):
                        self.category_history.append('power_chain')
                    
                    logger.info(f"✅ Generated feedback: '{response_text}' (praise: {should_praise})")
                    return response_text.strip()
                else:
                    if attempt < max_retries:
                        logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} - variety check failed")
                        prompt += "\nUse different wording and a different opening verb."
                    else:
                        logger.error("❌ Max retries reached, using fallback")
                        return "Adjust distance, stay active."
            
            except Exception as e:
                logger.error(f"❌ Gemini error on attempt {attempt + 1}: {e}")
                
                # Handle 429 rate limit specifically
                if "429" in str(e) or "rate limit" in str(e).lower():
                    # Parse retry delay from error if possible
                    retry_delay = 12.0  # Default 12 seconds
                    if "retry_delay" in str(e):
                        try:
                            import re
                            match = re.search(r'retry_delay[:\s]*(\d+(?:\.\d+)?)', str(e))
                            if match:
                                retry_delay = float(match.group(1))
                        except:
                            pass
                    
                    # Add jitter (±30%)
                    jitter = random.uniform(0.7, 1.3)
                    self.backoff_until = current_time + (retry_delay * jitter)
                    logger.warning(f"⏰ Rate limited, backoff until {self.backoff_until:.1f}s")
                    return "Shadowbox four beats while I reconnect."
                
                if attempt < max_retries:
                    continue
                else:
                    return "Tech glitch—shadowbox four beats while I reconnect."
        
        return "Adjust distance, stay active."
    
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
    
    def _blocking_stream_once(self, prompt: str, generation_config: dict) -> str:
        """Fully consume stream in this thread - no generator crosses await boundary"""
        try:
            # Generate content with streaming
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            # Consume all chunks in this thread
            text = ""
            for chunk in response:
                if getattr(chunk, "text", None):
                    text += chunk.text
            
            # SDK recommends resolving at the end of iteration
            try:
                response.resolve()
            except Exception:
                pass  # Ignore resolve errors
                
            return text.strip()  # Return plain string, never a generator
            
        except Exception as e:
            logger.error(f"❌ Blocking stream error: {e}")
            raise

    async def _generate_streaming_feedback(self, prompt: str) -> str:
        """Generate feedback using gemini-2.5-flash (final mode uses non-stream to avoid StopIteration)"""
        try:
            # Use gemini-2.5-flash for high RPM micro-cues
            generation_config = {
                'temperature': 0.8,  # Balanced creativity and consistency
                'max_output_tokens': 32,  # Keep it short for micro-cues
                'top_p': 0.9
            }
            
            # Final mode uses non-stream to avoid StopIteration/uvloop weirdness
            def _blocking_once():
                resp = self.model.generate_content(prompt, generation_config=generation_config, safety_settings=self._get_safety_settings())
                txt = self._extract_plain_text(resp)  # Use the safe helper
                return txt
            
            # Run non-streaming call in executor
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(None, _blocking_once)
            
            if response_text:
                # Enforce 15 word limit
                words = response_text.split()
                if len(words) > 15:
                    response_text = ' '.join(words[:15])
                
                logger.info(f"🔥 {self.model.model_name} response: '{response_text}'")
                return response_text.strip()
            else:
                # Soft fallback instead of raising — avoid WS error path
                logger.error(f"❌ Empty response from {self.model.model_name}; returning neutral cue")
                return "Keep moving, stay focused."
                
        except Exception as e:
            logger.error(f"❌ {self.model.model_name} error: {e}")
            # Soft fallback to keep session alive
            return "Keep moving, stay focused."
    

    
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
    
    def _extract_plain_text(self, resp) -> str:
        """
        Safely flatten Gemini responses to plain text.
        - Ignores non-text parts
        - Skips SAFETY-stopped candidates
        - Returns '' on failure
        """
        try:
            parts_out = []
            for c in getattr(resp, "candidates", []) or []:
                # If SDK exposes finish_reason, skip hard-stopped candidates
                fr = getattr(c, "finish_reason", None)
                if isinstance(fr, str) and fr.upper() == "SAFETY":
                    continue
                content = getattr(c, "content", None)
                if content is None:
                    continue
                for p in getattr(content, "parts", []) or []:
                    t = getattr(p, "text", None)
                    if t:
                        parts_out.append(t)
            return " ".join(parts_out).strip()
        except Exception:
            logger.exception("extract_plain_text failed")
            return ""
    
    def _safe_tokenize(self, chunk_text: str) -> list[str]:
        """Split on whitespace, keep punctuation on the word but use a clean compare"""
        safe = []
        for raw in chunk_text.split():
            cleaned = raw.lower().strip(",.!?;:\"'()[]{}")
            if cleaned in self.BANNED:
                continue  # skip banned nicknames mid-stream
            safe.append(raw)
        return safe
    
    def _extract_chunk_text(self, chunk) -> str:
        """Safely flatten a streaming chunk to text (handles candidates/parts)."""
        try:
            parts_out = []
            for c in getattr(chunk, "candidates", []) or []:
                content = getattr(c, "content", None)
                if content is None:
                    continue
                for p in getattr(content, "parts", []) or []:
                    t = getattr(p, "text", None)
                    if t:
                        parts_out.append(t)
            if parts_out:
                return " ".join(parts_out)
            # fallback to common SDK convenience
            t = getattr(chunk, "text", None)
            return t or ""
        except Exception:
            logger.exception("extract_chunk_text failed")
            return ""
    
    def _token_producer(self, prompt: str, generation_config: dict, q: Queue, stop_evt: threading.Event):
        """Producer (worker thread) — consume SDK stream safely"""
        try:
            # Contain the iterator in this thread; uvloop never sees it
            stream = self.model.generate_content(prompt, generation_config=generation_config, stream=True)
            for chunk in stream:
                if stop_evt.is_set():
                    break
                text = self._extract_chunk_text(chunk)
                if not text:
                    continue
                # Push raw text; async side will tokenize/filter
                q.put(text)
        except StopIteration:
            # Normal end-of-stream in some SDK internals — swallow it
            pass
        except Exception as e:
            q.put(e)  # signal failure to async side
        finally:
            q.put(_SENTINEL)
    
    async def _generate_token_stream(self, prompt: str, send_delta, send_final) -> str:
        """
        - send_delta: async function(str) -> None    (stream partial tokens to WS)
        - send_final: async function(str) -> None    (send final sanitized sentence)
        Returns final sanitized sentence (for TTS), or "" on failure.
        """
        generation_config = {'temperature': 0.8, 'max_output_tokens': 32, 'top_p': 0.9}
        q: Queue = Queue(maxsize=64)
        stop_evt = threading.Event()
        t = threading.Thread(target=self._token_producer, args=(prompt, generation_config, q, stop_evt), daemon=True)
        t.start()

        assembled = []
        loop = asyncio.get_running_loop()
        tokens_emitted = 0
        start_time = time.time()

        try:
            while True:
                item = await loop.run_in_executor(None, q.get)  # don't block event loop
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    # Soft fail: stop streaming; caller will fallback line
                    logger.warning(f"token stream error: {item}")
                    return ""
                # Tokenize + filter banned terms so nicknames never render
                tokens = self._safe_tokenize(str(item))
                if not tokens:
                    continue
                delta = " ".join(tokens)
                assembled.extend(tokens)
                tokens_emitted += 1
                # Stream delta to client
                await send_delta(delta)
        finally:
            stop_evt.set()

        # Build final sentence, sanitize, trim to ≤15 words, variety check
        final_text = " ".join(assembled).strip()
        if not final_text:
            return ""

        # Enforce 15-word limit
        words = final_text.split()
        if len(words) > 15:
            final_text = " ".join(words[:15])

        # Final sanitation pass (in case punctuation hid a banned term)
        sanitized = self._sanitize_response(final_text) or ""
        if not sanitized:
            return ""

        duration = time.time() - start_time
        logger.info(f"🔥 Token stream completed: {tokens_emitted} tokens, {duration:.2f}s duration")
        if tokens_emitted == 0:
            logger.warning("⚠️ Token stream produced zero tokens; likely SAFETY or non-text parts")

        await send_final(sanitized)
        return sanitized
    
    async def generate_elite_coaching_feedback_with_stream(self, comprehensive_analysis: Dict, send_delta, send_final) -> str:
        """Generate coaching feedback with token streaming support"""
        # Build the same prompt as the regular method
        import time
        
        # Handle special cases first
        if comprehensive_analysis.get("insufficient_data"):
            await send_final("Show me your stance and start moving.")
            return "Show me your stance and start moving."
        
        if comprehensive_analysis.get("no_pose_detected"):
            await send_final("Step back into frame—camera can't see you.")
            return "Step back into frame—camera can't see you."
        
        # Rate limiting check
        current_time = time.time()
        if current_time < self.backoff_until:
            logger.info(f"⏰ Rate limit backoff active until {self.backoff_until:.1f}s")
            await send_final("Shadowbox four beats while I reconnect.")
            return "Shadowbox four beats while I reconnect."
        
        if current_time - self.last_request_time < self.rate_limit_delay:
            logger.info(f"⏰ Rate limit delay: {self.rate_limit_delay - (current_time - self.last_request_time):.1f}s remaining")
            await send_final("Keep moving, stay focused.")
            return "Keep moving, stay focused."
        
        # Update metric history
        stance_data = comprehensive_analysis.get('stance_analysis', {})
        guard_data = comprehensive_analysis.get('guard_analysis', {})
        footwork_data = comprehensive_analysis.get('footwork_analysis', {})
        head_data = comprehensive_analysis.get('head_movement_analysis', {})
        
        current_metrics = {
            'hand_height': guard_data.get('hand_height', 0.0),
            'stance_width_ratio': stance_data.get('stance_width_ratio', 0.0),
            'total_movement': footwork_data.get('total_movement', 0.0),
            'head_mobility': head_data.get('head_movement_frequency', 0.0)
        }
        
        for metric, value in current_metrics.items():
            self.metric_history[metric].append(value)
        
        # Determine if we should praise
        should_praise = self._should_praise(current_metrics)
        
        # Build the new coaching prompt
        excluded_categories = list(self.category_history)
        jitter = random.randint(1000, 9999)
        
        # Get last opening verb for variety enforcement
        last_opening_verb = self.verb_history[-1] if self.verb_history else "None"
        
        prompt = f"""ROLE
You are an elite boxing coach delivering live micro-cues. Be urgent, specific, and professional.

HARD BANS
Never use address terms or nicknames: kid, bro, champ, buddy, pal, dude, man.
No profanity. No hashtags, emojis, or filler ("come on", "let's go", "you got this").

OUTPUT FORMAT
Return ONE sentence, plain text only (no quotes, no colons, no code blocks), ≤ 15 words.
Start with a direct, actionable cue (imperative or concise declarative). Avoid hedging.

VARIETY & REPETITION
Do NOT reuse the same opening word as the previous tip: {last_opening_verb}.
Avoid any bigram that appeared in the last 90 seconds (the app enforces; you must try).
If you cannot be fresh, pick a different focus area (see rotation) or rephrase.

FOCUS ROTATION (skip any in ExcludedCategories)
1) Guard (hands, elbows, chin)
2) Footwork (stance width, pivots, lateral movement)
3) Head movement (slips, rolls, angles)
4) Punch mechanics (hip rotation, snap, retraction)
5) Rhythm/breathing (tempo, relaxation)
6) Defense & counters (blocks, parries, timing)
7) Power chain (ground force, core engagement)

POSITIVE REINFORCEMENT (subtle, optional)
If a metric is above target or improving ≥10% vs. 30-frame average, prefix with 1–2 words of praise
(e.g., "Nice guard." "Good rhythm.") then the cue. Use praise at most 1 in 4 tips.

LIVE CONTEXT (use these numbers to choose and phrase the cue)
StanceWidthRatio: {current_metrics['stance_width_ratio']:.2f}   (target 0.55–0.70)
GuardHeight:     {current_metrics['hand_height']:.2f}         (higher is better)
FootworkActivity:{current_metrics['total_movement']:.2f}
HeadMobility:    {current_metrics['head_mobility']:.2f}
ExcludedCategories: {excluded_categories}
ShouldPraise: {should_praise}

FAIL-SAFES (only emit these exact lines when true)
- If no pose detected for >3s: Step back into frame—camera can't see you.
- If insufficient data (<5 frames): Show me your stance and start moving.

TONE EXAMPLES (style only; do not reuse wording)
"Tuck elbows; bring rear hand to cheek."
"Angle off right, double jab—don't square up."
"Good rhythm; add a slip after the cross."
"Rotate hip through, snap and retract fast."

RESPONSE
Output only the final one-sentence cue (≤15 words), plain text, nothing else.
PROMPT_JITTER_{jitter}"""
        
        # Update request time
        self.last_request_time = current_time
        
        # Generate with token streaming
        final_sentence = await self._generate_token_stream(prompt, send_delta, send_final)
        
        if final_sentence:
            # Add to history for variety checking
            self.last_5_responses.append(final_sentence)
            verb = self._get_first_verb(final_sentence)
            if verb:
                self.verb_history.append(verb)
            
            # Add category to history
            if any(word in final_sentence.lower() for word in ['chin', 'guard', 'hands', 'elbow']):
                self.category_history.append('guard')
            elif any(word in final_sentence.lower() for word in ['foot', 'step', 'move', 'bounce']):
                self.category_history.append('footwork')
            elif any(word in final_sentence.lower() for word in ['head', 'slip', 'duck']):
                self.category_history.append('head_movement')
            elif any(word in final_sentence.lower() for word in ['punch', 'jab', 'cross', 'hook']):
                self.category_history.append('punch_mechanics')
            elif any(word in final_sentence.lower() for word in ['rhythm', 'breathe', 'tempo']):
                self.category_history.append('rhythm')
            elif any(word in final_sentence.lower() for word in ['block', 'parry', 'counter']):
                self.category_history.append('defense')
            elif any(word in final_sentence.lower() for word in ['power', 'hip', 'core']):
                self.category_history.append('power_chain')
            
            logger.info(f"✅ Generated streaming feedback: '{final_sentence}' (praise: {should_praise})")
            return final_sentence
        else:
            # Fallback to non-streaming
            logger.warning("🔄 Token streaming failed, falling back to non-streaming")
            response_text = await self._generate_streaming_feedback(prompt)
            await send_final(response_text)
            return response_text
    
    def _get_safety_settings(self):
        """Return permissive safety settings to reduce empty SAFETY-stopped responses."""
        return [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
