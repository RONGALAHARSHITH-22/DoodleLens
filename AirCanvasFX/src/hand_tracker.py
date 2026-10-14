"""
Hand Tracker Module - DoodleLens
Detects hand landmarks using MediaPipe and provides smoothed fingertip positions.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict


class HandTracker:
    """Tracks hand landmarks and provides smoothed fingertip positions."""

    # MediaPipe landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_PIP = 6
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    PINKY_TIP = 20
    PINKY_PIP = 18

    def __init__(self, max_hands: int = 1,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5,
                 smoothing_factor: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.smoothing_factor = smoothing_factor
        self._smoothed_pos: Optional[np.ndarray] = None
        self._trail: deque = deque(maxlen=60)
        self.landmarks: Optional[object] = None
        self.hand_world_landmarks: Optional[object] = None
        self.handedness: str = "Unknown"
        self.frame_h: int = 480
        self.frame_w: int = 640

    def process(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        Process a frame and return hand tracking data.
        Returns None if no hand detected.
        """
        self.frame_h, self.frame_w = frame_rgb.shape[:2]
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            self._smoothed_pos = None
            self.landmarks = None
            return None

        # Use first detected hand
        hand_lm = results.multi_hand_landmarks[0]
        self.landmarks = hand_lm

        if results.multi_handedness:
            self.handedness = results.multi_handedness[0].classification[0].label

        # Get landmark positions
        lm_array = self._landmarks_to_array(hand_lm)

        # Index fingertip (landmark 8) - mirror the x coordinate
        index_raw = lm_array[self.INDEX_TIP]

        # Apply adaptive velocity-based smoothing:
        # High speed -> near-instant response (zero lag); slow/still -> smooth jitter reduction
        if self._smoothed_pos is None:
            self._smoothed_pos = index_raw.copy()
        else:
            dist = float(np.linalg.norm(index_raw - self._smoothed_pos))
            # Dynamic blend: dist >= 18px means fast drawing stroke -> alpha ~ 0.95 (instant)
            # dist <= 2px means stationary/aiming -> alpha ~ 0.40 (smooth out webcam jitter)
            speed_ratio = min(1.0, max(0.0, (dist - 2.0) / 16.0))
            alpha = 0.40 + 0.55 * speed_ratio
            self._smoothed_pos = (1.0 - alpha) * self._smoothed_pos + alpha * index_raw

        pos_px = self._smoothed_pos.astype(int)
        self._trail.append(tuple(pos_px))

        return {
            "index_tip": pos_px,
            "index_tip_raw": index_raw.astype(int),
            "landmarks_array": lm_array,
            "handedness": self.handedness,
            "trail": list(self._trail),
            "landmarks_obj": hand_lm,
        }

    def _landmarks_to_array(self, hand_landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to pixel coordinate array matching frame."""
        arr = np.zeros((21, 2), dtype=float)
        for i, lm in enumerate(hand_landmarks.landmark):
            arr[i] = [lm.x * self.frame_w, lm.y * self.frame_h]
        return arr

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw skeleton overlay on frame (optional debug view)."""
        if self.landmarks is None:
            return frame
        # Manually draw to match mirrored coordinates
        lm_arr = self._landmarks_to_array(self.landmarks)
        connections = self.mp_hands.HAND_CONNECTIONS
        for start_idx, end_idx in connections:
            p1 = tuple(lm_arr[start_idx].astype(int))
            p2 = tuple(lm_arr[end_idx].astype(int))
            cv2.line(frame, p1, p2, (0, 200, 100), 1)
        for i, pt in enumerate(lm_arr):
            cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 180), -1)
        return frame

    def get_finger_states(self, lm_array: np.ndarray) -> Dict[str, bool]:
        """Return a dict of which fingers are extended (rotation-invariant)."""
        if lm_array is None or len(lm_array) < 21:
            return {}

        wrist = lm_array[self.WRIST]

        def is_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            # Check Euclidean distances: extended finger tip is further from wrist and MCP than PIP
            d_tip_wrist = np.linalg.norm(lm_array[tip_idx] - wrist)
            d_pip_wrist = np.linalg.norm(lm_array[pip_idx] - wrist)
            d_tip_mcp = np.linalg.norm(lm_array[tip_idx] - lm_array[mcp_idx])
            d_pip_mcp = np.linalg.norm(lm_array[pip_idx] - lm_array[mcp_idx])

            # Invariant check: tip is further from wrist and extended away from MCP
            extended_dist = (d_tip_wrist > d_pip_wrist) and (d_tip_mcp > d_pip_mcp * 1.05)
            # Upright vertical fallback (tip higher than pip on screen)
            extended_vert = (lm_array[tip_idx][1] < lm_array[pip_idx][1]) and (d_tip_mcp > d_pip_mcp * 0.9)
            return bool(extended_dist or extended_vert)

        # Thumb extension: tip-pinky distance vs IP-pinky distance, or horizontal offset
        d_thumb_pinky = np.linalg.norm(lm_array[self.THUMB_TIP] - lm_array[17])
        d_thumb_ip_pinky = np.linalg.norm(lm_array[3] - lm_array[17])
        thumb_extended = (d_thumb_pinky > d_thumb_ip_pinky * 1.1) or (lm_array[self.THUMB_TIP][0] < lm_array[3][0])

        return {
            "thumb": bool(thumb_extended),
            "index": is_extended(self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            "middle": is_extended(self.MIDDLE_TIP, self.MIDDLE_PIP, 9),
            "ring": is_extended(self.RING_TIP, 14, 13),
            "pinky": is_extended(self.PINKY_TIP, self.PINKY_PIP, 17),
        }

    def get_pinch_distance(self, lm_array: np.ndarray) -> float:
        """Return normalized distance between thumb and index tips (0=pinch, 1=open)."""
        if lm_array is None or len(lm_array) < 21:
            return 1.0
        thumb = lm_array[self.THUMB_TIP]
        index = lm_array[self.INDEX_TIP]
        dist = np.linalg.norm(thumb - index)
        # Normalize by hand size (wrist to middle base)
        hand_size = np.linalg.norm(lm_array[self.WRIST] - lm_array[9]) + 1e-6
        return float(dist / hand_size)

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
