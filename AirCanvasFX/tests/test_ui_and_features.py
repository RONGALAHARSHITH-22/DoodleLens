"""
Tests for UI, settings persistence, shape snapping, and full integration.
"""

import sys
import os
import shutil
import tempfile
import numpy as np
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import AppController, AppMode
from src.drawing_engine import DrawingEngine, BrushType
from src.face_filter_engine import FilterType
from src.shape_recognition import ShapeRecognizer, ShapeType


class TestSettingsAndFeatures(unittest.TestCase):
    """Test settings persistence and new features."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.settings_file = os.path.join(self.tmpdir, "settings.json")
        self.controller = AppController(
            settings_path=self.settings_file,
            base_dir=self.tmpdir
        )

    def tearDown(self):
        self.controller.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_reload_settings(self):
        self.controller.camera.device_index = 2
        self.controller.camera.target_width = 1920
        self.controller.camera.target_height = 1080
        self.controller.auto_snap_shapes = True
        self.controller.show_fps = False
        saved = self.controller.save_settings()
        self.assertTrue(saved)
        self.assertTrue(os.path.exists(self.settings_file))

        # Reload controller from saved file
        reloaded = AppController(settings_path=self.settings_file, base_dir=self.tmpdir)
        self.assertEqual(reloaded.camera.device_index, 2)
        self.assertEqual(reloaded.camera.target_width, 1920)
        self.assertEqual(reloaded.camera.target_height, 1080)
        self.assertTrue(reloaded.auto_snap_shapes)
        self.assertFalse(reloaded.show_fps)
        reloaded.shutdown()

    def test_shape_clean_points(self):
        rec = ShapeRecognizer()

        # Circle
        circle_pts = [(int(100 + 50 * np.cos(t)), int(100 + 50 * np.sin(t)))
                      for t in np.linspace(0, 2 * np.pi, 30)]
        pts = rec.get_clean_points(ShapeType.CIRCLE, circle_pts)
        self.assertIsNotNone(pts)
        self.assertGreater(len(pts), 10)

        # Line
        line_pts = [(0, 0), (50, 50), (100, 100)]
        pts = rec.get_clean_points(ShapeType.LINE, line_pts)
        self.assertIsNotNone(pts)
        self.assertEqual(pts, [(0, 0), (100, 100)])

        # Unknown
        self.assertIsNone(rec.get_clean_points(ShapeType.UNKNOWN, line_pts))

    def test_replace_last_stroke_points(self):
        engine = DrawingEngine(640, 480)
        engine.start_stroke()
        engine.add_point((10, 10))
        engine.add_point((20, 20))
        engine.end_stroke()
        self.assertEqual(len(engine.strokes), 1)

        clean = [(0, 0), (100, 100)]
        engine.replace_last_stroke_points(clean)
        self.assertEqual(engine.strokes[-1].points, clean)
        self.assertEqual(len(engine.strokes), 1)

    def test_drawing_grayscale_load(self):
        """Ensure loading a 2D grayscale image does not throw IndexError."""
        import cv2
        gray_path = os.path.join(self.tmpdir, "test_gray.png")
        gray_img = np.full((100, 100), 200, dtype=np.uint8)
        cv2.imwrite(gray_path, gray_img)

        engine = DrawingEngine(320, 240)
        loaded = engine.load_drawing(gray_path)
        self.assertTrue(loaded)
        self.assertEqual(engine.canvas.shape, (240, 320, 4))


class TestMainWindowUI(unittest.TestCase):
    """Test PyQt6 MainWindow in offscreen mode."""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication, QMessageBox
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
        cls._orig_info = QMessageBox.information
        cls._orig_question = QMessageBox.question
        cls._orig_warning = QMessageBox.warning
        cls._orig_critical = QMessageBox.critical
        QMessageBox.information = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
        QMessageBox.question = lambda *args, **kwargs: QMessageBox.StandardButton.Yes
        QMessageBox.warning = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
        QMessageBox.critical = lambda *args, **kwargs: QMessageBox.StandardButton.Ok

    @classmethod
    def tearDownClass(cls):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information = cls._orig_info
        QMessageBox.question = cls._orig_question
        QMessageBox.warning = cls._orig_warning
        QMessageBox.critical = cls._orig_critical

    def setUp(self):

        self.tmpdir = tempfile.mkdtemp()
        from src.ui.main_window import MainWindow
        self.win = MainWindow(base_dir=self.tmpdir)

    def tearDown(self):
        self.win.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_navigation_tabs(self):
        # 0: Drawing
        self.win._nav(0)
        self.assertEqual(self.win._cs.currentIndex(), 0)
        self.assertEqual(self.win.controller.mode, AppMode.DRAWING)

        # 1: Face Filter
        self.win._nav(1)
        self.assertEqual(self.win._cs.currentIndex(), 0)
        self.assertEqual(self.win.controller.mode, AppMode.FACE_FILTER)

        # 2: My Creations
        self.win._nav(2)
        self.assertEqual(self.win._cs.currentIndex(), 1)

        # 3: Settings
        self.win._nav(3)
        self.assertEqual(self.win._cs.currentIndex(), 2)

    def test_brush_and_color_controls(self):
        # Change brush
        self.win.brush_combo.setCurrentIndex(1)
        selected_brush = self.win.brush_combo.currentData()
        self.assertEqual(self.win.controller.drawing_engine.brush_type, selected_brush)

        # Change size
        self.win.size_sl.setValue(24)
        self.assertEqual(self.win.controller.drawing_engine.brush_size, 24)

        # Change glow
        self.win.glow_sl.setValue(8)
        self.assertEqual(self.win.controller.drawing_engine.glow_intensity, 8)

        # Toggle eraser
        self.win.eraser_btn.setChecked(True)
        self.assertEqual(self.win.controller.drawing_engine.brush_type, BrushType.ERASER)
        self.win.eraser_btn.setChecked(False)

        # Toggle snap shapes
        self.win.snap_btn.setChecked(True)
        self.assertTrue(self.win.controller.auto_snap_shapes)

    def test_filters_selection(self):
        self.win._sel_filter(FilterType.CYBERPUNK_MASK)
        self.assertEqual(self.win.controller.face_filter_engine.active_filter, FilterType.CYBERPUNK_MASK)
        self.assertEqual(self.win.controller.mode, AppMode.FACE_FILTER)

        # Reset to none
        self.win._sel_filter(FilterType.NONE)
        self.assertEqual(self.win.controller.face_filter_engine.active_filter, FilterType.NONE)
        self.assertEqual(self.win.controller.mode, AppMode.DRAWING)

    def test_creations_and_custom_filter(self):
        # Draw on canvas
        self.win.controller.drawing_engine.start_stroke()
        self.win.controller.drawing_engine.add_point((100, 100))
        self.win.controller.drawing_engine.add_point((200, 200))
        self.win.controller.drawing_engine.end_stroke()

        # Convert to filter
        ok = self.win.controller.convert_drawing_to_filter(name="my_test_mask", anchor="forehead")
        self.assertTrue(ok)
        self.win._refr_cust()
        self.win._refr_creations()

        # Custom filter list in combo
        names = [self.win.cust_combo.itemText(i) for i in range(self.win.cust_combo.count())]
        self.assertIn("my_test_mask", names)

    def test_apply_settings_in_window(self):
        self.win._sc_idx.setValue(1)
        self.win._sc_w.setValue(800)
        self.win._sc_h.setValue(600)
        self.win._ssnap.setChecked(True)
        self.win._apply_settings()

        self.assertEqual(self.win.controller.camera.device_index, 1)
        self.assertEqual(self.win.controller.camera.target_width, 800)
        self.assertEqual(self.win.controller.camera.target_height, 600)
        self.assertTrue(self.win.controller.auto_snap_shapes)


if __name__ == "__main__":
    unittest.main()
