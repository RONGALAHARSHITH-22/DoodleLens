"""
Face Tracker Module - DoodleLens
Tracks face landmarks using MediaPipe Face Mesh with smoothing.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple, List


# Key facial landmark indices for MediaPipe Face Mesh (468 landmarks)
FACE_LANDMARKS = {
    "nose_tip": 4,
    "nose_bridge": 6,
    "left_eye_center": 468,   # iris center (only if using iris landmarks)
    "right_eye_center": 473,
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_eyebrow_inner": 107,
    "left_eyebrow_outer": 46,
    "right_eyebrow_inner": 336,
    "right_eyebrow_outer": 276,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "chin": 152,
    "forehead": 10,
    "left_cheek": 234,
    "right_cheek": 454,
    "face_top": 10,
    "face_bottom": 152,
    "face_left": 234,
    "face_right": 454,
}

# Simpler landmark set that doesn't use iris (safer compatibility)
SAFE_LANDMARKS = {
    "nose_tip": 4,
    "nose_bridge": 6,
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_eyebrow_outer": 46,
    "right_eyebrow_outer": 276,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "chin": 152,
    "forehead": 10,
    "left_cheek": 234,
    "right_cheek": 454,
}


class FaceTracker:
    """Tracks facial landmarks using MediaPipe Face Mesh."""

    def __init__(self, detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 smoothing_factor: float = 0.6):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.smoothing_factor = smoothing_factor

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # skip iris for compatibility
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.frame_h: int = 480
        self.frame_w: int = 640

        # Smoothed landmark storage
        self._smoothed_landmarks: Optional[np.ndarray] = None
        self.landmarks_array: Optional[np.ndarray] = None
        self.face_bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h

    def process(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        Process a frame and return face tracking data.
        Returns None if no face detected.
        """
        self.frame_h, self.frame_w = frame_rgb.shape[:2]
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self._smoothed_landmarks = None
            self.landmarks_array = None
            self.face_bbox = None
            return None

        face_lm = results.multi_face_landmarks[0]

        # Convert to pixel array matching frame
        raw = np.zeros((468, 2), dtype=float)
        for i, lm in enumerate(face_lm.landmark):
            raw[i] = [lm.x * self.frame_w, lm.y * self.frame_h]

        # Temporal smoothing
        if self._smoothed_landmarks is None:
            self._smoothed_landmarks = raw.copy()
        else:
            self._smoothed_landmarks = (
                self.smoothing_factor * self._smoothed_landmarks +
                (1 - self.smoothing_factor) * raw
            )

        self.landmarks_array = self._smoothed_landmarks.astype(int)

        # Compute face bounding box
        xs = self.landmarks_array[:, 0]
        ys = self.landmarks_array[:, 1]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        self.face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Extract named landmarks
        named = {}
        for name, idx in SAFE_LANDMARKS.items():
            if idx < len(self.landmarks_array):
                named[name] = tuple(self.landmarks_array[idx])

        # Compute additional derived points
        if "left_eye_inner" in named and "left_eye_outer" in named:
            le = named["left_eye_inner"]
            lo = named["left_eye_outer"]
            named["left_eye_center"] = (
                (le[0] + lo[0]) // 2,
                (le[1] + lo[1]) // 2
            )
        if "right_eye_inner" in named and "right_eye_outer" in named:
            re = named["right_eye_inner"]
            ro = named["right_eye_outer"]
            named["right_eye_center"] = (
                (re[0] + ro[0]) // 2,
                (re[1] + ro[1]) // 2
            )

        # Face size metrics
        face_w = self.face_bbox[2]
        face_h = self.face_bbox[3]
        face_cx = self.face_bbox[0] + face_w // 2
        face_cy = self.face_bbox[1] + face_h // 2

        # Head tilt estimate (angle of line between eye centers)
        head_tilt = 0.0
        if "left_eye_center" in named and "right_eye_center" in named:
            lec = named["left_eye_center"]
            rec = named["right_eye_center"]
            dx = rec[0] - lec[0]
            dy = rec[1] - lec[1]
            head_tilt = float(np.degrees(np.arctan2(dy, dx)))

        return {
            "landmarks": self.landmarks_array,
            "named_landmarks": named,
            "face_bbox": self.face_bbox,
            "face_center": (face_cx, face_cy),
            "face_width": face_w,
            "face_height": face_h,
            "head_tilt": head_tilt,
        }

    def draw_mesh_overlay(self, frame: np.ndarray,
                          color: Tuple = (0, 200, 100),
                          draw_contours_only: bool = True) -> np.ndarray:
        """Draw face mesh or contours on frame."""
        if self.landmarks_array is None:
            return frame

        if draw_contours_only:
            # Draw key points
            for name, idx in SAFE_LANDMARKS.items():
                if idx < len(self.landmarks_array):
                    pt = tuple(self.landmarks_array[idx])
                    cv2.circle(frame, pt, 2, color, -1)
        else:
            for pt in self.landmarks_array:
                cv2.circle(frame, tuple(pt), 1, color, -1)

        if self.face_bbox:
            x, y, w, h = self.face_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 200), 1)

        return frame

    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
