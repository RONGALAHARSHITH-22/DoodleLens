"""
Tests - DoodleLens
Unit tests for core modules.
"""

import sys
import os
import numpy as np
import unittest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGestureRecognition(unittest.TestCase):
    """Test gesture classification logic."""

    def setUp(self):
        from src.gesture_recognition import GestureRecognizer, Gesture
        self.rec = GestureRecognizer(debounce_frames=1, confidence_threshold=0.5)
        self.Gesture = Gesture

    def test_index_extended(self):
        states = {"thumb": False, "index": True, "middle": False,
                  "ring": False, "pinky": False}
        g = self.rec.classify(states, 0.8)
        self.assertEqual(g, self.Gesture.INDEX_EXTENDED)

    def test_open_palm(self):
        states = {"thumb": True, "index": True, "middle": True,
                  "ring": True, "pinky": True}
        g = self.rec.classify(states, 0.8)
        self.assertEqual(g, self.Gesture.OPEN_PALM)

    def test_closed_fist(self):
        states = {"thumb": False, "index": False, "middle": False,
                  "ring": False, "pinky": False}
        g = self.rec.classify(states, 0.8)
        self.assertEqual(g, self.Gesture.CLOSED_FIST)

    def test_thumbs_up(self):
        states = {"thumb": True, "index": False, "middle": False,
                  "ring": False, "pinky": False}
        g = self.rec.classify(states, 0.8)
        self.assertEqual(g, self.Gesture.THUMBS_UP)

    def test_two_fingers(self):
        states = {"thumb": False, "index": True, "middle": True,
                  "ring": False, "pinky": False}
        g = self.rec.classify(states, 0.8)
        self.assertEqual(g, self.Gesture.TWO_FINGERS)

    def test_pinch(self):
        states = {"thumb": True, "index": True, "middle": False,
                  "ring": False, "pinky": False}
        g = self.rec.classify(states, 0.1)  # close pinch
        self.assertEqual(g, self.Gesture.PINCH)

    def test_empty_states(self):
        g = self.rec.classify({}, 0.8)
        self.assertEqual(g, self.Gesture.UNKNOWN)

    def test_debouncing(self):
        from src.gesture_recognition import GestureRecognizer, Gesture
        rec = GestureRecognizer(debounce_frames=5, confidence_threshold=0.8)
        states = {"thumb": False, "index": True, "middle": False,
                  "ring": False, "pinky": False}
        # Should require multiple consistent frames
        for _ in range(5):
            rec.update(states, 0.8)
        self.assertEqual(rec.current_gesture, Gesture.INDEX_EXTENDED)


class TestDrawingEngine(unittest.TestCase):
    """Test drawing engine behavior."""

    def setUp(self):
        from src.drawing_engine import DrawingEngine, BrushType
        self.engine = DrawingEngine(width=640, height=480)
        self.BrushType = BrushType

    def test_start_stroke(self):
        self.engine.start_stroke()
        self.assertTrue(self.engine.is_drawing)
        self.assertIsNotNone(self.engine.current_stroke)

    def test_add_point(self):
        self.engine.start_stroke()
        self.engine.add_point((100, 100))
        self.engine.add_point((150, 150))
        self.assertEqual(len(self.engine.current_stroke.points), 2)

    def test_end_stroke(self):
        self.engine.start_stroke()
        self.engine.add_point((100, 100))
        self.engine.end_stroke()
        self.assertFalse(self.engine.is_drawing)
        self.assertEqual(len(self.engine.strokes), 1)

    def test_undo_redo(self):
        self.engine.start_stroke()
        self.engine.add_point((100, 100))
        self.engine.end_stroke()
        self.assertEqual(len(self.engine.strokes), 1)

        self.engine.undo()
        self.assertEqual(len(self.engine.strokes), 0)

        self.engine.redo()
        self.assertEqual(len(self.engine.strokes), 1)

    def test_clear(self):
        self.engine.start_stroke()
        self.engine.add_point((100, 100))
        self.engine.end_stroke()
        self.engine.clear()
        self.assertEqual(len(self.engine.strokes), 0)

    def test_canvas_shape(self):
        self.assertEqual(self.engine.canvas.shape, (480, 640, 4))

    def test_out_of_bounds_points(self):
        """Points outside canvas should not crash."""
        self.engine.start_stroke()
        self.engine.add_point((-10, -10))
        self.engine.add_point((9999, 9999))
        self.engine.end_stroke()

    def test_brush_types(self):
        for brush in self.BrushType:
            self.engine.set_brush(brush)
            self.engine.start_stroke()
            self.engine.add_point((320, 240))
            self.engine.end_stroke()

    def test_composite_frame(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = self.engine.get_composite_frame(dummy_frame)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_tracking_loss_no_connect(self):
        """Points should not connect after tracking loss (new stroke each time)."""
        self.engine.start_stroke()
        self.engine.add_point((50, 50))
        self.engine.end_stroke()

        # Simulate tracking loss
        self.engine.start_stroke()
        self.engine.add_point((500, 400))  # Far away point
        self.engine.end_stroke()

        self.assertEqual(len(self.engine.strokes), 2)
        # Second stroke should start at (500,400), not connect to (50,50)
        pts_1 = self.engine.strokes[0].points
        pts_2 = self.engine.strokes[1].points
        self.assertNotEqual(pts_1[-1], pts_2[0])

    def test_save_load(self):
        from src.drawing_engine import DrawingEngine
        import tempfile
        self.engine.start_stroke()
        self.engine.add_point((100, 100))
        self.engine.add_point((200, 200))
        self.engine.end_stroke()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            saved = self.engine.save_drawing(path)
            self.assertTrue(saved)
            new_engine = DrawingEngine(width=640, height=480)
            loaded = new_engine.load_drawing(path)
            self.assertTrue(loaded)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestShapeRecognition(unittest.TestCase):
    """Test shape recognition from stroke points."""

    def setUp(self):
        from src.shape_recognition import ShapeRecognizer, ShapeType
        self.rec = ShapeRecognizer()
        self.ShapeType = ShapeType

    def _make_circle_pts(self, cx=320, cy=240, r=100, n=60):
        import math
        return [(int(cx + r * math.cos(2 * math.pi * i / n)),
                 int(cy + r * math.sin(2 * math.pi * i / n)))
                for i in range(n + 1)]

    def _make_line_pts(self, n=40):
        # Generate a perfectly straight horizontal line with many points
        return [(10 + i * 8, 100) for i in range(n)]

    def test_circle_recognition(self):
        pts = self._make_circle_pts()
        result = self.rec.recognize(pts)
        self.assertEqual(result, self.ShapeType.CIRCLE)

    def test_line_recognition(self):
        pts = self._make_line_pts()
        result = self.rec.recognize(pts)
        self.assertEqual(result, self.ShapeType.LINE)

    def test_too_few_points(self):
        result = self.rec.recognize([(0, 0), (10, 10)])
        self.assertEqual(result, self.ShapeType.UNKNOWN)


class TestCustomFilterManager(unittest.TestCase):
    """Test save/load of custom filters."""

    def setUp(self):
        import tempfile
        from src.custom_filter_manager import CustomFilterManager
        self.tmpdir = tempfile.mkdtemp()
        self.mgr = CustomFilterManager(save_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_and_load(self):
        # Create a simple BGRA canvas with a line drawn
        canvas = np.zeros((480, 640, 4), dtype=np.uint8)
        import cv2
        cv2.line(canvas, (100, 100), (300, 300), (255, 0, 255, 255), 5)

        cf = self.mgr.create_filter_from_canvas(
            name="test_filter",
            canvas_bgra=canvas,
            anchor="forehead",
            scale=1.0,
            rotation=0.0,
            offset=(0, 0),
        )
        self.assertIsNotNone(cf)
        self.assertEqual(cf.name, "test_filter")
        self.assertEqual(cf.anchor, "forehead")

        # Load it back
        overlay = self.mgr.load_overlay("test_filter")
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.ndim, 3)
        self.assertEqual(overlay.shape[2], 4)

    def test_list_filters(self):
        canvas = np.zeros((480, 640, 4), dtype=np.uint8)
        import cv2
        cv2.circle(canvas, (320, 240), 50, (0, 255, 255, 255), -1)
        self.mgr.create_filter_from_canvas("filter_a", canvas)
        self.mgr.create_filter_from_canvas("filter_b", canvas)
        names = [f.name for f in self.mgr.list_filters()]
        self.assertIn("filter_a", names)
        self.assertIn("filter_b", names)

    def test_delete_filter(self):
        canvas = np.zeros((480, 640, 4), dtype=np.uint8)
        import cv2
        cv2.circle(canvas, (100, 100), 30, (255, 100, 0, 255), -1)
        self.mgr.create_filter_from_canvas("to_delete", canvas)
        self.assertTrue(self.mgr.delete_filter("to_delete"))
        self.assertIsNone(self.mgr.get_filter("to_delete"))

    def test_empty_canvas_returns_none(self):
        canvas = np.zeros((480, 640, 4), dtype=np.uint8)  # Fully transparent
        result = self.mgr.create_filter_from_canvas("empty", canvas)
        self.assertIsNone(result)

    def test_persistence(self):
        """Saved filters should reload after manager restart."""
        from src.custom_filter_manager import CustomFilterManager
        canvas = np.zeros((480, 640, 4), dtype=np.uint8)
        import cv2
        cv2.rectangle(canvas, (50, 50), (200, 200), (100, 255, 100, 255), -1)
        self.mgr.create_filter_from_canvas("persistent", canvas)

        # Reload manager
        new_mgr = CustomFilterManager(save_dir=self.tmpdir)
        self.assertIsNotNone(new_mgr.get_filter("persistent"))


class TestCoordinateMapping(unittest.TestCase):
    """Test coordinate mirroring and mapping."""

    def test_mirror_x(self):
        """Landmarks should have mirrored x coordinate."""
        frame_w = 640
        # If landmark normalized x = 0.2 (left of frame)
        # After mirroring: (1 - 0.2) * 640 = 512 (right)
        raw_x = 0.2
        mirrored_x = (1.0 - raw_x) * frame_w
        self.assertAlmostEqual(mirrored_x, 512.0)

    def test_canvas_bounds(self):
        from src.drawing_engine import DrawingEngine
        engine = DrawingEngine(640, 480)
        # Out of bounds points should not crash
        engine.start_stroke()
        for pt in [(-100, -100), (800, 600), (0, 0), (639, 479)]:
            engine.add_point(pt)
        engine.end_stroke()
        self.assertEqual(engine.canvas.shape, (480, 640, 4))


class TestHandTrackerImprovements(unittest.TestCase):
    """Test rotation-invariant finger states and adaptive tracking."""

    def setUp(self):
        from src.hand_tracker import HandTracker
        self.tracker = HandTracker()

    def test_upright_index_finger_extended(self):
        # Create a mock 21-landmark array for upright hand pointing up
        lm = np.zeros((21, 2), dtype=float)
        # Wrist at bottom center
        lm[0] = [320, 400]
        # Index: MCP at [320, 300], PIP at [320, 250], TIP at [320, 180]
        lm[5] = [320, 300]
        lm[6] = [320, 250]
        lm[8] = [320, 180]
        # Other fingers curled (tip closer to wrist than PIP)
        for tip, pip, mcp in [(12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            lm[mcp] = [350, 320]
            lm[pip] = [350, 340]
            lm[tip] = [350, 380] # curled into palm
        lm[4] = [280, 380] # thumb curled
        lm[3] = [280, 360]

        states = self.tracker.get_finger_states(lm)
        self.assertTrue(states["index"])
        self.assertFalse(states["middle"])
        self.assertFalse(states["ring"])
        self.assertFalse(states["pinky"])

    def test_horizontal_index_finger_extended(self):
        # Pointing horizontally to the right (y is identical, only x differs)
        lm = np.zeros((21, 2), dtype=float)
        lm[0] = [200, 240] # wrist
        lm[5] = [280, 240] # index MCP
        lm[6] = [340, 240] # index PIP
        lm[8] = [420, 240] # index TIP (further from wrist in X, identical Y)
        # Other fingers curled back toward wrist
        for tip, pip, mcp in [(12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            lm[mcp] = [270, 260]
            lm[pip] = [290, 260]
            lm[tip] = [250, 260] # curled inward
        lm[4] = [220, 230]
        lm[3] = [230, 230]

        states = self.tracker.get_finger_states(lm)
        self.assertTrue(states["index"], "Index pointing horizontally should register as extended")
        self.assertFalse(states["middle"])
        self.assertFalse(states["ring"])
        self.assertFalse(states["pinky"])

    def test_hysteresis_and_stroke_continuity(self):
        from src.gesture_recognition import GestureRecognizer, Gesture
        rec = GestureRecognizer(debounce_frames=2, confidence_threshold=0.5)

        # 1. Start drawing
        states = {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}
        rec.update(states, 0.8)
        rec.update(states, 0.8)
        self.assertEqual(rec.current_gesture, Gesture.INDEX_EXTENDED)

        # 2. Transient noisy/unknown frame should NOT immediately abort INDEX_EXTENDED
        rec.update({}, 0.8)
        self.assertEqual(rec.current_gesture, Gesture.INDEX_EXTENDED)

        # 3. Sustained different gesture (fist) should cleanly update
        fist = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
        rec.update(fist, 0.8)
        rec.update(fist, 0.8)
        self.assertEqual(rec.current_gesture, Gesture.CLOSED_FIST)


if __name__ == "__main__":
    print("🧪 Running DoodleLens Tests")
    print("=" * 50)
    unittest.main(verbosity=2)
