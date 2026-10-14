"""
Camera Manager - DoodleLens
Handles webcam capture, threading, and frame delivery.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple


class CameraManager:
    """
    Manages webcam capture in a dedicated thread.
    Delivers frames to a callback without blocking the UI.
    """

    def __init__(self,
                 device_index: int = 0,
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 60):
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self._error_callback: Optional[Callable[[str], None]] = None

        # Stats
        self.fps_actual: float = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()

    def set_frame_callback(self, cb: Callable[[np.ndarray], None]):
        """Register callback called with each new frame (from capture thread)."""
        self._frame_callback = cb

    def set_error_callback(self, cb: Callable[[str], None]):
        """Register callback called when a camera error occurs."""
        self._error_callback = cb

    def start(self, device_index: Optional[int] = None) -> bool:
        """Open the webcam and start capturing. Returns True on success."""
        if self._running:
            return True

        if device_index is not None:
            self.device_index = device_index

        candidate_indices = [self.device_index]
        for idx in range(3):
            if idx not in candidate_indices:
                candidate_indices.append(idx)

        cap = None
        used_idx = self.device_index

        for idx in candidate_indices:
            for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                try:
                    c = cv2.VideoCapture(idx, api)
                    if c.isOpened():
                        # Uncap 60 FPS on Windows USB cameras via MJPG format
                        try:
                            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        except Exception:
                            pass
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                        c.set(cv2.CAP_PROP_FPS, self.target_fps)
                        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, test_frame = c.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            cap = c
                            used_idx = idx
                            break
                        else:
                            c.release()
                    else:
                        c.release()
                except Exception:
                    pass
            if cap is not None:
                break

        if cap is None:
            self._notify_error(f"Cannot open any camera device (tried indices {candidate_indices})")
            return False

        self.device_index = used_idx


        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                         daemon=True, name="CameraCapture")
        self._thread.start()
        return True


    def stop(self):
        """Stop capturing and release the webcam."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._latest_frame = None

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._notify_error("Camera disconnected")
                break

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Mirror frame (webcam default is not mirrored)
            frame = cv2.flip(frame, 1)

            with self._lock:
                self._latest_frame = frame

            # FPS calculation
            self._fps_counter += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self.fps_actual = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = now

            if self._frame_callback is not None:
                try:
                    self._frame_callback(frame)
                except Exception:
                    pass

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recently captured frame (thread-safe)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_resolution(self) -> Tuple[int, int]:
        """Return actual camera resolution (width, height)."""
        if self._cap is not None and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self.target_width, self.target_height)

    def _notify_error(self, msg: str):
        if self._error_callback:
            self._error_callback(msg)

    @property
    def is_running(self) -> bool:
        return self._running

    def list_cameras(self) -> list:
        """Try to list available camera indices (0–4)."""
        available = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
