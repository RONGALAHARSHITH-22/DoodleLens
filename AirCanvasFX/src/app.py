"""
App Controller - DoodleLens
Orchestrates all modules: camera, tracking, drawing, face filters, UI.
"""

import cv2
import numpy as np
import os
import json
import time
import threading
from typing import Optional, Dict, Tuple
from datetime import datetime

from .camera_manager import CameraManager
from .hand_tracker import HandTracker
from .gesture_recognition import GestureRecognizer, Gesture
from .drawing_engine import DrawingEngine, BrushType
from .face_tracker import FaceTracker
from .face_filter_engine import FaceFilterEngine, FilterType
from .custom_filter_manager import CustomFilterManager, ANCHOR_OPTIONS
from .shape_recognition import ShapeRecognizer, ShapeType


class AppMode:
    DRAWING = "drawing"
    FACE_FILTER = "face_filter"


class AppController:
    """
    Central controller connecting camera, hand/face tracking,
    drawing engine, face filters, and the UI.
    """

    def __init__(self, settings_path: str = "config/settings.json",
                 base_dir: str = "."):
        self.base_dir = base_dir
        self.settings_path = settings_path
        self.settings = self._load_settings(settings_path)
        self.mode = AppMode.DRAWING
        self.auto_snap_shapes = self.settings.get("drawing", {}).get("auto_snap_shapes", False)
        self.last_detected_shape: Optional[str] = None
        self._drawing_grace_count: int = 0
        self._hand_lost_count: int = 0
        self.MAX_STROKE_GRACE: int = 5

        # -- Camera
        cam_cfg = self.settings.get("camera", {})
        self.camera = CameraManager(
            device_index=cam_cfg.get("device_index", 0),
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 60),
        )

        track_cfg = self.settings.get("tracking", {})

        # -- Hand tracker
        self.hand_tracker = HandTracker(
            max_hands=track_cfg.get("max_hands", 1),
            detection_confidence=track_cfg.get("hand_detection_confidence", 0.7),
            tracking_confidence=track_cfg.get("hand_tracking_confidence", 0.5),
            smoothing_factor=track_cfg.get("smoothing_factor", 0.4),
        )

        # -- Gesture recognizer
        gest_cfg = self.settings.get("gestures", {})
        self.gesture_rec = GestureRecognizer(
            debounce_frames=gest_cfg.get("debounce_frames", 2),
            confidence_threshold=gest_cfg.get("confidence_threshold", 0.5),
        )

        # -- Drawing engine
        draw_cfg = self.settings.get("drawing", {})
        perf_cfg = self.settings.get("performance", {})
        self.drawing_engine = DrawingEngine(
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            max_particles=perf_cfg.get("max_particles", 200),
        )
        self._init_drawing_defaults(draw_cfg)

        # -- Face tracker
        self.face_tracker = FaceTracker(
            detection_confidence=track_cfg.get("face_detection_confidence", 0.5),
            tracking_confidence=track_cfg.get("face_tracking_confidence", 0.5),
            smoothing_factor=track_cfg.get("smoothing_factor", 0.7),
        )

        # -- Face filter engine
        self.face_filter_engine = FaceFilterEngine(
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
        )

        # -- Custom filter manager
        custom_dir = os.path.join(base_dir, "custom_filters")
        self.custom_filter_manager = CustomFilterManager(save_dir=custom_dir)

        # -- Shape recognizer
        self.shape_recognizer = ShapeRecognizer()

        # -- State
        self.is_paused = False
        self.show_landmarks = self.settings.get("ui", {}).get("show_landmarks", False)
        self.show_fps = self.settings.get("ui", {}).get("show_fps", True)
        self.show_gesture = self.settings.get("ui", {}).get("show_gesture_indicator", True)

        self.hand_data: Optional[Dict] = None
        self.face_data: Optional[Dict] = None
        self.current_gesture = Gesture.UNKNOWN
        self.gesture_label = ""

        self._saved_drawings_dir = os.path.join(base_dir, "saved_drawings")
        os.makedirs(self._saved_drawings_dir, exist_ok=True)

        # Callbacks for UI updates
        self.on_frame_ready: Optional[callable] = None
        self.on_status_update: Optional[callable] = None
        self.on_gesture_change: Optional[callable] = None
        self.on_error: Optional[callable] = None

        # Processing lock
        self._proc_lock = threading.Lock()

        # Register gesture callbacks
        self.gesture_rec.register_callback(Gesture.THUMBS_UP, self._on_snapshot_gesture)
        self.gesture_rec.register_callback(Gesture.TWO_FINGERS, self._on_mode_switch)
        self.gesture_rec.register_callback(Gesture.CLOSED_FIST, self._on_clear_gesture)

        self._last_mode_switch = 0.0

    def _load_settings(self, path: str) -> Dict:
        full_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path
        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _init_drawing_defaults(self, draw_cfg: Dict):
        brush_name = draw_cfg.get("default_brush", "neon_pen")
        try:
            self.drawing_engine.brush_type = BrushType(brush_name)
        except ValueError:
            self.drawing_engine.brush_type = BrushType.NEON_PEN
        color = draw_cfg.get("default_color", [0, 255, 255])
        self.drawing_engine.brush_color = tuple(color)
        self.drawing_engine.brush_size = draw_cfg.get("brush_size", 8)
        self.drawing_engine.glow_intensity = draw_cfg.get("glow_intensity", 5)
        self.drawing_engine.opacity = draw_cfg.get("opacity", 1.0)

    def start_camera(self, device_index: Optional[int] = None) -> bool:
        self.camera.set_frame_callback(self._process_frame)
        self.camera.set_error_callback(self._on_camera_error)
        return self.camera.start(device_index)


    def stop_camera(self):
        self.camera.stop()

    def _process_frame(self, frame: np.ndarray):
        """Called from camera thread for each new frame."""
        if self.is_paused:
            if self.on_frame_ready:
                self.on_frame_ready(frame, None, None)
            return

        with self._proc_lock:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hand tracking
            self.hand_data = self.hand_tracker.process(frame_rgb)

            # Face tracking (only run when needed for filter or skeleton overlay to save 25-40ms per frame)
            needs_face = (
                self.mode == AppMode.FACE_FILTER or
                (hasattr(self, "face_filter_engine") and self.face_filter_engine.active_filter != FilterType.NONE) or
                self.show_landmarks
            )
            if needs_face:
                self.face_data = self.face_tracker.process(frame_rgb)
            else:
                self.face_data = None

            # Gesture recognition & stroke continuity
            if self.hand_data:
                self._hand_lost_count = 0
                finger_states = self.hand_tracker.get_finger_states(
                    self.hand_data["landmarks_array"]
                )
                pinch_dist = self.hand_tracker.get_pinch_distance(
                    self.hand_data["landmarks_array"]
                )
                self.current_gesture = self.gesture_rec.update(finger_states, pinch_dist)
                self.gesture_label = self.gesture_rec.get_label()
            else:
                self._hand_lost_count += 1
                if self.drawing_engine.is_drawing:
                    if self._hand_lost_count > self.MAX_STROKE_GRACE:
                        self.drawing_engine.end_stroke()
                        self._snap_last_shape_if_enabled()
                self.current_gesture = Gesture.UNKNOWN
                self.gesture_label = ""

            # Drawing control
            if self.mode == AppMode.DRAWING:
                self._handle_drawing_gesture()

            # Particle update
            self.drawing_engine.update_particles()

            # Compose output frame
            out = self._compose_frame(frame)

            if self.on_frame_ready:
                self.on_frame_ready(out, self.hand_data, self.face_data)

    def _handle_drawing_gesture(self):
        """Translate current gesture into drawing commands with stroke continuity."""
        gesture = self.current_gesture
        hand_data = self.hand_data

        if gesture == Gesture.INDEX_EXTENDED and hand_data:
            self._drawing_grace_count = 0
            tip = hand_data["index_tip"]
            if not self.drawing_engine.is_drawing:
                self.drawing_engine.start_stroke()
            self.drawing_engine.add_point(tuple(tip))
        elif self.drawing_engine.is_drawing:
            # Explicit intention to cancel or switch
            explicit_stop = gesture in (Gesture.OPEN_PALM, Gesture.CLOSED_FIST, Gesture.PINCH, Gesture.TWO_FINGERS)
            if explicit_stop:
                self._drawing_grace_count = 0
                self.drawing_engine.end_stroke()
                self._snap_last_shape_if_enabled()
            else:
                # Transient uncertainty or quick hand movement: hold stroke open for grace frames
                self._drawing_grace_count += 1
                if self._drawing_grace_count > self.MAX_STROKE_GRACE:
                    self._drawing_grace_count = 0
                    self.drawing_engine.end_stroke()
                    self._snap_last_shape_if_enabled()

    def _snap_last_shape_if_enabled(self):
        """Snap last drawn stroke to a geometric shape if auto-snap is enabled."""
        if self.auto_snap_shapes and self.drawing_engine.strokes:
            last_stroke = self.drawing_engine.strokes[-1]
            if len(last_stroke.points) >= 10:
                shape = self.shape_recognizer.recognize(last_stroke.points)
                if shape != ShapeType.UNKNOWN:
                    clean_pts = self.shape_recognizer.get_clean_points(shape, last_stroke.points)
                    if clean_pts:
                        self.drawing_engine.replace_last_stroke_points(clean_pts)
                        self.last_detected_shape = shape.value.title()
                        if self.on_status_update:
                            self.on_status_update(f"✨ Snapped Shape: {self.last_detected_shape}")


    def _compose_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compose the final display frame with all overlays."""
        out = frame.copy()

        # Face filter first (background effect)
        if self.face_data:
            out = self.face_filter_engine.render(out, self.face_data)

        # Drawing overlay
        out = self.drawing_engine.get_composite_frame(out)

        # Landmark overlays (debug)
        if self.show_landmarks:
            out = self.hand_tracker.draw_landmarks(out)
            out = self.face_tracker.draw_mesh_overlay(out)

        # Fingertip cursor
        if self.hand_data:
            tip = tuple(self.hand_data["index_tip"])
            if self.current_gesture == Gesture.INDEX_EXTENDED:
                color = list(self.drawing_engine.brush_color)
                cv2.circle(out, tip, self.drawing_engine.brush_size + 4, color, 2)
                cv2.circle(out, tip, 3, (255, 255, 255), -1)
            else:
                cv2.circle(out, tip, 8, (200, 200, 200), 1)

        # HUD
        self._draw_hud(out)

        return out

    def _draw_hud(self, frame: np.ndarray):
        """Draw on-screen HUD: FPS, gesture, mode, status."""
        h, w = frame.shape[:2]

        # FPS
        if self.show_fps:
            fps_text = f"FPS: {self.camera.fps_actual:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        # Mode
        mode_text = f"Mode: {self.mode.replace('_', ' ').title()}"
        cv2.putText(frame, mode_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        # Gesture
        if self.show_gesture and self.gesture_label:
            cv2.putText(frame, self.gesture_label, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Hand tracking status
        hand_status = "Hand: ✓" if self.hand_data else "Hand: —"
        face_status = "Face: ✓" if self.face_data else "Face: —"
        cv2.putText(frame, f"{hand_status}  {face_status}",
                    (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # -- Gesture callbacks --------------------------------------------------

    def _on_snapshot_gesture(self):
        """Called when thumbs-up gesture is held."""
        self.take_snapshot()

    def _on_mode_switch(self):
        """Switch between drawing and face-filter modes (debounced)."""
        now = time.time()
        if now - self._last_mode_switch < 1.5:
            return
        self._last_mode_switch = now
        if self.mode == AppMode.DRAWING:
            self.mode = AppMode.FACE_FILTER
        else:
            self.mode = AppMode.DRAWING
        if self.on_status_update:
            self.on_status_update(f"Mode: {self.mode}")

    def _on_clear_gesture(self):
        """Clear canvas on closed-fist (configurable)."""
        self.drawing_engine.clear()

    # -- Public API --------------------------------------------------------

    def take_snapshot(self) -> Optional[str]:
        """Save current composed frame as PNG."""
        frame = self.camera.get_latest_frame()
        if frame is None:
            return None
        composed = self._compose_frame(frame)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._saved_drawings_dir, f"snapshot_{ts}.png")
        cv2.imwrite(path, composed)
        return path

    def save_drawing(self, filename: Optional[str] = None) -> Optional[str]:
        """Save the current drawing canvas."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawing_{ts}.png"
        path = os.path.join(self._saved_drawings_dir, filename)
        if self.drawing_engine.save_drawing(path):
            return path
        return None

    def convert_drawing_to_filter(self, name: str, anchor: str = "forehead",
                                    scale: float = 1.0, rotation: float = 0.0,
                                    offset: Tuple[int, int] = (0, 0)) -> bool:
        """Convert the current drawing into a saved custom face filter."""
        canvas = self.drawing_engine.get_canvas_image()
        cf = self.custom_filter_manager.create_filter_from_canvas(
            name=name,
            canvas_bgra=canvas,
            anchor=anchor,
            scale=scale,
            rotation=rotation,
            offset=offset,
        )
        return cf is not None

    def apply_custom_filter(self, filter_name: str):
        """Load and apply a custom filter."""
        overlay = self.custom_filter_manager.load_overlay(filter_name)
        cf = self.custom_filter_manager.get_filter(filter_name)
        if overlay is not None and cf is not None:
            self.face_filter_engine.set_custom_overlay(
                overlay=overlay,
                anchor=cf.anchor,
                scale=cf.scale,
                rotation=cf.rotation,
                offset=(cf.offset_x, cf.offset_y),
            )

    def undo(self):
        self.drawing_engine.undo()

    def redo(self):
        self.drawing_engine.redo()

    def clear_canvas(self):
        self.drawing_engine.clear()

    def set_brush(self, brush: BrushType):
        self.drawing_engine.set_brush(brush)

    def set_color(self, color: Tuple[int, int, int]):
        self.drawing_engine.set_color(color)

    def set_brush_size(self, size: int):
        self.drawing_engine.set_size(size)

    def set_glow(self, glow: int):
        self.drawing_engine.set_glow(glow)

    def set_face_filter(self, filter_type: FilterType):
        self.face_filter_engine.set_filter(filter_type)
        if self.mode != AppMode.FACE_FILTER:
            self.mode = AppMode.FACE_FILTER

    def set_filter_intensity(self, intensity: float):
        self.face_filter_engine.filter_intensity = max(0.0, min(2.0, intensity))

    def _on_camera_error(self, msg: str):
        if self.on_error:
            self.on_error(msg)

    def save_settings(self, path: Optional[str] = None) -> bool:
        """Save current configuration to JSON file."""
        target_path = path or self.settings_path
        if not target_path:
            return False

        self.settings.setdefault("camera", {})["device_index"] = self.camera.device_index
        self.settings["camera"]["width"] = self.camera.target_width
        self.settings["camera"]["height"] = self.camera.target_height
        self.settings["camera"]["fps"] = self.camera.target_fps

        self.settings.setdefault("tracking", {})["smoothing_factor"] = self.hand_tracker.smoothing_factor
        self.settings.setdefault("performance", {})["max_particles"] = self.drawing_engine.max_particles
        self.settings.setdefault("ui", {})["show_fps"] = self.show_fps
        self.settings["ui"]["show_landmarks"] = self.show_landmarks
        self.settings["ui"]["show_gesture_indicator"] = self.show_gesture
        self.settings.setdefault("gestures", {})["debounce_frames"] = self.gesture_rec.debounce_frames
        self.settings["gestures"]["confidence_threshold"] = self.gesture_rec.confidence_threshold
        self.settings.setdefault("drawing", {})["auto_snap_shapes"] = self.auto_snap_shapes

        full_path = os.path.join(self.base_dir, target_path) if not os.path.isabs(target_path) else target_path
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception:
            return False

    def shutdown(self):
        """Cleanly release all resources."""
        self.stop_camera()
        self.hand_tracker.release()
        self.face_tracker.release()

