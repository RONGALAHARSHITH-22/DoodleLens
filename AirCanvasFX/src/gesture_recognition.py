"""
Gesture Recognition Module - DoodleLens
Classifies hand gestures from landmark arrays with debouncing.
"""

import numpy as np
from collections import deque
from typing import Optional, Dict
from enum import Enum, auto


class Gesture(Enum):
    UNKNOWN = auto()
    INDEX_EXTENDED = auto()   # Drawing mode
    OPEN_PALM = auto()        # Pause / move
    CLOSED_FIST = auto()      # Clear / pause
    PINCH = auto()            # Tool select
    THUMBS_UP = auto()        # Snapshot / confirm
    TWO_FINGERS = auto()      # Switch mode
    PEACE = auto()            # Peace sign


GESTURE_LABELS = {
    Gesture.UNKNOWN: "Unknown",
    Gesture.INDEX_EXTENDED: "✏️ Drawing",
    Gesture.OPEN_PALM: "✋ Pause",
    Gesture.CLOSED_FIST: "✊ Clear",
    Gesture.PINCH: "🤌 Select",
    Gesture.THUMBS_UP: "👍 Snapshot",
    Gesture.TWO_FINGERS: "✌️ Switch Mode",
    Gesture.PEACE: "✌️ Peace",
}


class GestureRecognizer:
    """
    Recognizes hand gestures using finger states and pinch distance.
    Applies debouncing to avoid accidental activations.
    """

    def __init__(self, debounce_frames: int = 2,
                 confidence_threshold: float = 0.5,
                 pinch_threshold: float = 0.35):
        self.debounce_frames = debounce_frames
        self.confidence_threshold = confidence_threshold
        self.pinch_threshold = pinch_threshold

        self._history: deque = deque(maxlen=debounce_frames)
        self._current_gesture: Gesture = Gesture.UNKNOWN
        self._gesture_frame_count: int = 0
        self._last_confirmed: Gesture = Gesture.UNKNOWN

        # Track last activated gestures for callback-style usage
        self._gesture_callbacks: Dict[Gesture, callable] = {}

    def classify(self, finger_states: Dict[str, bool],
                 pinch_dist: float) -> Gesture:
        """Classify gesture from finger states and pinch distance."""
        if not finger_states:
            return Gesture.UNKNOWN

        thumb = finger_states.get("thumb", False)
        index = finger_states.get("index", False)
        middle = finger_states.get("middle", False)
        ring = finger_states.get("ring", False)
        pinky = finger_states.get("pinky", False)

        extended = [thumb, index, middle, ring, pinky]
        count = sum(extended)

        # Pinch: thumb + index close together, other fingers down
        if pinch_dist < self.pinch_threshold and not middle and not ring and not pinky:
            return Gesture.PINCH

        # All fingers extended = open palm
        if count >= 4:
            return Gesture.OPEN_PALM

        # Thumbs up: only thumb extended
        if thumb and not index and not middle and not ring and not pinky:
            return Gesture.THUMBS_UP

        # Closed fist - no fingers extended
        if count == 0:
            return Gesture.CLOSED_FIST

        # Two fingers (index + middle, peace/switch mode)
        if index and middle and not ring and not pinky:
            if thumb:
                return Gesture.TWO_FINGERS  # Three with thumb = still two mode
            return Gesture.TWO_FINGERS

        # Only index extended = draw (thumb position can vary during drawing)
        if index and not middle and not ring and not pinky:
            return Gesture.INDEX_EXTENDED

        return Gesture.UNKNOWN

    def update(self, finger_states: Dict[str, bool],
               pinch_dist: float) -> Gesture:
        """
        Update gesture state with debouncing and drawing hysteresis.
        Returns the debounced (stable) gesture.
        """
        raw = self.classify(finger_states, pinch_dist)
        self._history.append(raw)

        # Majority vote over history
        if len(self._history) < self.debounce_frames:
            # If fast-tracking index extended during drawing
            if raw == Gesture.INDEX_EXTENDED and self._current_gesture == Gesture.INDEX_EXTENDED:
                self._gesture_frame_count += 1
                return self._current_gesture
            return self._current_gesture

        counts: Dict[Gesture, int] = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1

        best = max(counts, key=lambda g: counts[g])
        best_ratio = counts[best] / len(self._history)

        if best_ratio >= self.confidence_threshold:
            if best != self._current_gesture:
                self._current_gesture = best
                self._gesture_frame_count = 0
                # Fire callback if registered
                if best in self._gesture_callbacks:
                    self._gesture_callbacks[best]()
            else:
                self._gesture_frame_count += 1
        elif self._current_gesture == Gesture.INDEX_EXTENDED and raw in (Gesture.INDEX_EXTENDED, Gesture.UNKNOWN):
            # Drawing hysteresis: keep drawing alive across momentary uncertain frames
            self._gesture_frame_count += 1

        return self._current_gesture

    def register_callback(self, gesture: Gesture, callback: callable):
        """Register a callback triggered when a gesture is first detected."""
        self._gesture_callbacks[gesture] = callback

    def get_label(self) -> str:
        """Return human-readable label of current gesture."""
        return GESTURE_LABELS.get(self._current_gesture, "Unknown")

    @property
    def current_gesture(self) -> Gesture:
        return self._current_gesture

    def reset(self):
        """Reset internal state."""
        self._history.clear()
        self._current_gesture = Gesture.UNKNOWN
        self._gesture_frame_count = 0
