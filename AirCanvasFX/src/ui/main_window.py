"""
Main Window - DoodleLens
PyQt6 main application window — full-featured with all panels.
"""

import sys
import os
import cv2
import numpy as np
from typing import Optional
import threading

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QColorDialog, QFileDialog,
    QFrame, QScrollArea, QInputDialog, QMessageBox, QSizePolicy,
    QGridLayout, QGroupBox, QStackedWidget, QStatusBar,
    QButtonGroup, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFormLayout, QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QKeySequence, QShortcut, QIcon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.app import AppController, AppMode
from src.drawing_engine import BrushType, BRUSH_NAMES
from src.face_filter_engine import FilterType, FILTER_NAMES
from src.custom_filter_manager import ANCHOR_OPTIONS

# ── Style tokens ──────────────────────────────────────────────────────────────
DARK_BG        = "#09090f"
PANEL_BG       = "#111118"
CARD_BG        = "#1a1a24"
CARD_HOVER     = "#222232"
BORDER         = "#252535"
BORDER_HI      = "#3a3a55"
ACCENT_PURPLE  = "#9b5cf6"
ACCENT_CYAN    = "#06b6d4"
ACCENT_MAGENTA = "#e11d96"
ACCENT_GREEN   = "#10b981"
ACCENT_ORANGE  = "#f97316"
TEXT_PRIMARY   = "#f0f0ff"
TEXT_SECONDARY = "#7070a0"
TEXT_MUTED     = "#404060"


def _mk_style():
    """Build the global stylesheet string."""
    s = []
    s.append(f"QMainWindow,QWidget{{background:{DARK_BG};color:{TEXT_PRIMARY};"
              f"font-family:'Segoe UI',Arial,sans-serif;font-size:13px;}}")
    s.append(f"QLabel{{color:{TEXT_PRIMARY};background:transparent;}}")
    s.append(
        f"QPushButton{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        f"stop:0 #222232,stop:1 #181826);color:{TEXT_PRIMARY};"
        f"border:1px solid {BORDER};border-radius:7px;padding:6px 14px;min-height:30px;}}"
    )
    s.append(
        f"QPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        f"stop:0 #2e1e50,stop:1 #221840);border-color:{ACCENT_PURPLE};color:white;}}"
    )
    s.append(f"QPushButton:pressed{{background:#180f2a;border-color:{ACCENT_PURPLE};}}")
    s.append(
        f"QPushButton:checked{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        f"stop:0 {ACCENT_PURPLE},stop:1 #6d28d9);border-color:#c084fc;color:white;}}"
    )
    s.append(f"QPushButton:disabled{{background:{CARD_BG};color:{TEXT_MUTED};border-color:{BORDER};}}")
    s.append(
        f"QPushButton#accent{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
        f"stop:0 {ACCENT_PURPLE},stop:1 {ACCENT_CYAN});"
        f"color:white;font-weight:bold;border:none;border-radius:8px;}}"
    )
    s.append(
        f"QPushButton#accent:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
        f"stop:0 #a78bfa,stop:1 #22d3ee);}}"
    )
    s.append(f"QPushButton#danger{{background:#1a0808;border-color:#6b1111;color:#f87171;}}")
    s.append(f"QPushButton#danger:hover{{background:#350a0a;border-color:#ef4444;color:#fca5a5;}}")
    s.append(f"QPushButton#success{{background:#0a1a0e;border-color:#065f46;color:#6ee7b7;}}")
    s.append(f"QPushButton#success:hover{{background:#0d2818;border-color:{ACCENT_GREEN};color:#a7f3d0;}}")
    s.append(f"QSlider::groove:horizontal{{background:{BORDER};height:4px;border-radius:2px;}}")
    s.append(
        f"QSlider::handle:horizontal{{background:{ACCENT_PURPLE};width:14px;height:14px;"
        f"margin:-5px 0;border-radius:7px;border:2px solid #c084fc;}}"
    )
    s.append(f"QSlider::handle:horizontal:hover{{background:#c084fc;border-color:white;}}")
    s.append(
        f"QSlider::sub-page:horizontal{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
        f"stop:0 {ACCENT_PURPLE},stop:1 {ACCENT_CYAN});border-radius:2px;}}"
    )
    s.append(
        f"QComboBox{{background:{CARD_BG};border:1px solid {BORDER};"
        f"border-radius:6px;padding:5px 8px;color:{TEXT_PRIMARY};min-height:30px;}}"
    )
    s.append(f"QComboBox:hover{{border-color:{ACCENT_PURPLE};}}")
    s.append(
        f"QComboBox QAbstractItemView{{background:#1a1a28;border:1px solid {ACCENT_PURPLE};"
        f"color:{TEXT_PRIMARY};selection-background-color:{ACCENT_PURPLE};outline:none;}}"
    )
    s.append(f"QFrame#panel{{background:{PANEL_BG};border:1px solid {BORDER};border-radius:12px;}}")
    s.append(
        f"QGroupBox{{border:1px solid {BORDER};border-radius:7px;margin-top:10px;"
        f"font-weight:bold;color:{TEXT_SECONDARY};font-size:10px;"
        f"text-transform:uppercase;letter-spacing:1.2px;padding-top:4px;}}"
    )
    s.append("QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 6px;}")
    s.append("QScrollArea{border:none;background:transparent;}")
    s.append(f"QScrollBar:vertical{{background:{CARD_BG};width:5px;border-radius:3px;margin:0;}}")
    s.append(f"QScrollBar::handle:vertical{{background:{BORDER_HI};border-radius:3px;min-height:20px;}}")
    s.append(f"QScrollBar::handle:vertical:hover{{background:{ACCENT_PURPLE};}}")
    s.append("QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
    s.append(
        f"QStatusBar{{background:{PANEL_BG};color:{TEXT_SECONDARY};"
        f"border-top:1px solid {BORDER};font-size:11px;padding:0 8px;}}"
    )
    s.append(f"QCheckBox{{color:{TEXT_PRIMARY};spacing:8px;}}")
    s.append(
        f"QCheckBox::indicator{{width:16px;height:16px;border:1px solid {BORDER_HI};"
        f"border-radius:4px;background:{CARD_BG};}}"
    )
    s.append(f"QCheckBox::indicator:checked{{background:{ACCENT_PURPLE};border-color:#c084fc;}}")
    s.append(f"QCheckBox::indicator:hover{{border-color:{ACCENT_PURPLE};}}")
    s.append(
        f"QSpinBox,QDoubleSpinBox{{background:{CARD_BG};border:1px solid {BORDER};"
        f"border-radius:6px;padding:4px 8px;color:{TEXT_PRIMARY};min-height:28px;}}"
    )
    s.append(f"QSpinBox:hover,QDoubleSpinBox:hover{{border-color:{ACCENT_PURPLE};}}")
    s.append(
        f"QLineEdit{{background:{CARD_BG};border:1px solid {BORDER};"
        f"border-radius:6px;padding:5px 8px;color:{TEXT_PRIMARY};min-height:28px;}}"
    )
    s.append(f"QLineEdit:focus{{border-color:{ACCENT_PURPLE};}}")
    return "\n".join(s)


STYLESHEET = _mk_style()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sec_lbl(text, color=TEXT_SECONDARY):
    l = QLabel(text.upper())
    l.setStyleSheet(f"color:{color};font-size:9px;font-weight:bold;letter-spacing:1.8px;padding:6px 0 3px 0;")
    return l


def _glow_lbl(text, color=ACCENT_CYAN):
    l = QLabel(text)
    l.setStyleSheet(f"color:{color};font-size:16px;font-weight:bold;letter-spacing:1.5px;padding:2px 0;")
    return l


def _div():
    d = QFrame()
    d.setFrameShape(QFrame.Shape.HLine)
    d.setFixedHeight(1)
    d.setStyleSheet(f"background:{BORDER};border:none;margin:4px 0;")
    return d


# ─── Custom widgets ───────────────────────────────────────────────────────────

class ColorButton(QPushButton):
    colorChanged = pyqtSignal(tuple)

    def __init__(self, color=(0, 255, 255), parent=None):
        super().__init__(parent)
        self._c = color
        self.setFixedSize(36, 36)
        self.setToolTip("Click to pick brush color")
        self._refresh()
        self.clicked.connect(self._pick)

    def _refresh(self):
        r, g, b = self._c[2], self._c[1], self._c[0]
        self.setStyleSheet(
            f"QPushButton{{background:rgb({r},{g},{b});border:2px solid rgba(255,255,255,0.4);border-radius:18px;}}"
            f"QPushButton:hover{{border-color:{ACCENT_PURPLE};border-width:3px;}}"
        )

    def _pick(self):
        r, g, b = self._c[2], self._c[1], self._c[0]
        c = QColorDialog.getColor(QColor(r, g, b), self, "Pick Brush Color")
        if c.isValid():
            self._c = (c.blue(), c.green(), c.red())
            self._refresh()
            self.colorChanged.emit(self._c)

    def set_color(self, bgr):
        self._c = bgr
        self._refresh()

    def get_color(self):
        return self._c


class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            f"QLabel{{background:#040408;border:2px solid {BORDER};"
            f"border-radius:12px;color:{TEXT_SECONDARY};font-size:18px;}}"
        )
        self.setText("📷  Camera not started\n\nClick  ▶ Start Camera  to begin")
        self._last_pixmap: Optional[QPixmap] = None

    def set_frame(self, frame: np.ndarray):
        if frame is None:
            return
        frame = np.ascontiguousarray(frame)
        h, w, ch = frame.shape
        qi = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self._last_pixmap = QPixmap.fromImage(qi)
        self._update_display()

    def _update_display(self):
        if self._last_pixmap and not self._last_pixmap.isNull():
            target_w = max(self.width(), 640)
            target_h = max(self.height(), 480)
            scaled = self._last_pixmap.scaled(
                target_w, target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            if not scaled.isNull():
                self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()




class StatusDot(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._set("#444")

    def _set(self, color):
        self.setStyleSheet(
            f"QLabel{{background:{color};border-radius:5px;border:1px solid rgba(255,255,255,0.15);}}"
        )

    def set_ok(self):   self._set(ACCENT_GREEN)
    def set_off(self):  self._set("#444")
    def set_warn(self): self._set(ACCENT_ORANGE)


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self, base_dir="."):
        super().__init__()
        self.base_dir = base_dir
        self.setWindowTitle("DoodleLens — AI Drawing & Face Filter Studio")
        icon_path = os.path.join(base_dir, "assets", "icons", "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1280, 800)
        self.resize(1520, 950)
        self.setStyleSheet(STYLESHEET)

        self.controller = AppController(
            settings_path=os.path.join(base_dir, "config", "settings.json"),
            base_dir=base_dir,
        )
        self.controller.on_frame_ready = self._on_frame_ready
        self.controller.on_error = self._on_cam_error
        self.controller.on_status_update = self._on_status

        self._pending_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._build_ui()
        self._setup_shortcuts()

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._update_ui)
        self._timer.start()

        self.frame_received.connect(self._display_frame)

        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._sb.showMessage("Ready — Click '▶ Start Camera' or wait for auto-connect...")



    # ── Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        cen = QWidget()
        self.setCentralWidget(cen)
        root = QVBoxLayout(cen)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_nav())

        cont = QWidget()
        cl = QHBoxLayout(cont)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(10)

        self._lp = self._build_left()
        cl.addWidget(self._lp, 0)

        self._cs = QStackedWidget()
        self._cs.addWidget(self._build_cam_view())
        self._cs.addWidget(self._build_creations())
        self._cs.addWidget(self._build_settings())
        cl.addWidget(self._cs, 1)

        self._rp = self._build_right()
        cl.addWidget(self._rp, 0)

        root.addWidget(cont, 1)

    def _build_nav(self):
        nav = QFrame()
        nav.setObjectName("panel")
        nav.setFixedHeight(58)
        nav.setStyleSheet(
            f"QFrame{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #09090f,stop:0.35 #0f0a20,stop:0.65 #0f0a20,stop:1 #09090f);"
            f"border-bottom:1px solid {ACCENT_PURPLE};border-radius:0;}}"
        )
        ly = QHBoxLayout(nav)
        ly.setContentsMargins(20, 0, 20, 0)
        ly.setSpacing(4)

        logo = QLabel("✨ DoodleLens")
        logo.setStyleSheet(f"font-size:21px;font-weight:bold;color:#c084fc;letter-spacing:2px;padding-right:16px;")
        ly.addWidget(logo)

        self._dot = StatusDot()
        self._dot.setToolTip("Camera status")
        ly.addWidget(self._dot)
        ly.addSpacing(12)

        self._nb_grp = QButtonGroup(self)
        self._nb_grp.setExclusive(True)
        self._nbtns = []
        nstyle = (
            f"QPushButton{{background:transparent;border:1px solid transparent;border-radius:7px;"
            f"color:{TEXT_SECONDARY};font-size:13px;padding:4px 18px;font-weight:500;}}"
            f"QPushButton:hover{{color:{TEXT_PRIMARY};border-color:{BORDER_HI};background:rgba(255,255,255,0.04);}}"
            f"QPushButton:checked{{color:{ACCENT_CYAN};border-color:{ACCENT_CYAN};background:rgba(6,182,212,0.12);font-weight:bold;}}"
        )
        for i, lbl in enumerate(["✏️  Air Drawing", "🎭  Face Filters", "🖼️  My Creations", "⚙️  Settings"]):
            b = QPushButton(lbl)
            b.setCheckable(True)
            b.setFixedHeight(38)
            b.setStyleSheet(nstyle)
            b.clicked.connect(lambda _, idx=i: self._nav(idx))
            self._nb_grp.addButton(b, i)
            ly.addWidget(b)
            self._nbtns.append(b)
        self._nbtns[0].setChecked(True)

        ly.addStretch()

        hk_btn = QPushButton("⌨️  Shortcuts")
        hk_btn.setFixedHeight(30)
        hk_btn.setToolTip("View keyboard shortcuts & gestures [F1]")
        hk_btn.setStyleSheet(
            f"QPushButton{{background:{CARD_BG};border:1px solid {BORDER};border-radius:6px;"
            f"color:{TEXT_SECONDARY};font-size:11px;padding:3px 10px;}}"
            f"QPushButton:hover{{border-color:{ACCENT_PURPLE};color:{TEXT_PRIMARY};}}"
        )
        hk_btn.clicked.connect(self._show_shortcuts_dialog)
        ly.addWidget(hk_btn)
        ly.addSpacing(6)

        self._fps_lbl = QLabel("FPS —")
        self._fps_lbl.setStyleSheet(
            f"background:{CARD_BG};color:{ACCENT_GREEN};font-size:11px;font-weight:bold;"
            f"padding:3px 8px;border-radius:6px;border:1px solid #065f46;"
        )
        ly.addWidget(self._fps_lbl)

        ver = QLabel("v1.0")
        ver.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;padding-left:8px;")
        ly.addWidget(ver)
        return nav


    def _build_cam_view(self):
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(0, 0, 0, 0)
        ly.setSpacing(8)
        self.video_label = VideoLabel()
        ly.addWidget(self.video_label, 1)
        ly.addWidget(self._build_bottom())
        return w

    def _build_left(self):
        p = QFrame()
        p.setObjectName("panel")
        p.setFixedWidth(196)
        ly = QVBoxLayout(p)
        ly.setContentsMargins(10, 14, 10, 14)
        ly.setSpacing(5)

        ly.addWidget(_glow_lbl("🖌  Tools", ACCENT_PURPLE))
        ly.addWidget(_div())

        ly.addWidget(_sec_lbl("Brush Type"))
        self.brush_combo = QComboBox()
        self.brush_combo.setToolTip("Select brush style")
        for brush in BrushType:
            if brush != BrushType.ERASER:
                self.brush_combo.addItem(BRUSH_NAMES[brush], brush)
        self.brush_combo.currentIndexChanged.connect(self._on_brush_change)
        ly.addWidget(self.brush_combo)

        ly.addSpacing(2)
        ly.addWidget(_sec_lbl("Color"))
        crow = QHBoxLayout()
        crow.setSpacing(6)
        self.color_btn = ColorButton(color=(255, 255, 0))
        self.color_btn.colorChanged.connect(lambda c: self.controller.set_color(c))
        crow.addWidget(self.color_btn)

        pal_grid = QGridLayout()
        pal_grid.setSpacing(3)
        colors = [
            (255, 255, 0),    # Cyan
            (255, 0, 180),    # Neon Violet
            (180, 20, 255),   # Hot Pink
            (0, 140, 255),    # Sunset Orange
            (0, 235, 255),    # Solar Yellow
            (0, 255, 60),     # Lime Green
            (255, 120, 0),    # Electric Blue
            (255, 255, 255),  # Pure White
        ]
        for idx, bgr in enumerate(colors):
            pb = QPushButton()
            pb.setFixedSize(20, 20)
            r, g, b = bgr[2], bgr[1], bgr[0]
            pb.setToolTip(f"RGB({r},{g},{b})")
            pb.setStyleSheet(
                f"QPushButton{{background:rgb({r},{g},{b});border-radius:10px;border:1px solid rgba(255,255,255,0.25);}}"
                f"QPushButton:hover{{border:2px solid white;}}"
            )
            pb.clicked.connect(lambda chk, c=bgr: (self.color_btn.set_color(c), self.controller.set_color(c)))
            pal_grid.addWidget(pb, idx // 4, idx % 4)
        crow.addLayout(pal_grid)
        ly.addLayout(crow)

        ly.addSpacing(2)
        ly.addWidget(_sec_lbl("Brush Size"))
        self.size_sl = QSlider(Qt.Orientation.Horizontal)
        self.size_sl.setRange(1, 50); self.size_sl.setValue(8)
        self.size_sl.setToolTip("Brush size (1–50 px)  [ [ / ] ]")
        self.size_sl.valueChanged.connect(lambda v: self.controller.set_brush_size(v))
        ly.addWidget(self.size_sl)
        self._sz_lbl = QLabel("8 px")
        self._sz_lbl.setStyleSheet(f"color:{ACCENT_CYAN};font-size:11px;font-weight:bold;")
        self.size_sl.valueChanged.connect(lambda v: self._sz_lbl.setText(f"{v} px"))
        ly.addWidget(self._sz_lbl)

        ly.addWidget(_sec_lbl("Glow Intensity"))
        self.glow_sl = QSlider(Qt.Orientation.Horizontal)
        self.glow_sl.setRange(0, 10); self.glow_sl.setValue(5)
        self.glow_sl.setToolTip("Glow radius layers (0–10)")
        self.glow_sl.valueChanged.connect(lambda v: self.controller.set_glow(v))
        ly.addWidget(self.glow_sl)
        self._gl_lbl = QLabel("5")
        self._gl_lbl.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:11px;")
        self.glow_sl.valueChanged.connect(lambda v: self._gl_lbl.setText(str(v)))
        ly.addWidget(self._gl_lbl)

        ly.addWidget(_sec_lbl("Opacity"))
        self.opacity_sl = QSlider(Qt.Orientation.Horizontal)
        self.opacity_sl.setRange(10, 100); self.opacity_sl.setValue(100)
        self.opacity_sl.setToolTip("Brush opacity (10%–100%)")
        self.opacity_sl.valueChanged.connect(lambda v: setattr(self.controller.drawing_engine, "opacity", v / 100.0))
        ly.addWidget(self.opacity_sl)
        self._op_lbl = QLabel("100%")
        self._op_lbl.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:11px;")
        self.opacity_sl.valueChanged.connect(lambda v: self._op_lbl.setText(f"{v}%"))
        ly.addWidget(self._op_lbl)

        ly.addWidget(_div())

        er_row = QHBoxLayout(); er_row.setSpacing(5)
        self.eraser_btn = QPushButton("⬜ Eraser")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setToolTip("Toggle eraser mode  [E]")
        self.eraser_btn.toggled.connect(self._on_eraser)

        self.snap_btn = QPushButton("✨ Snap")
        self.snap_btn.setCheckable(True)
        self.snap_btn.setChecked(self.controller.auto_snap_shapes)
        self.snap_btn.setToolTip("Auto-snap geometric shapes (circles, lines, boxes)")
        self.snap_btn.toggled.connect(lambda v: setattr(self.controller, "auto_snap_shapes", v))

        er_row.addWidget(self.eraser_btn, 1)
        er_row.addWidget(self.snap_btn, 1)
        ly.addLayout(er_row)
        ly.addSpacing(2)


        br = QHBoxLayout(); br.setSpacing(5)
        self.undo_btn = QPushButton("↩ Undo")
        self.undo_btn.setToolTip("Undo last stroke  [Ctrl+Z]")
        self.undo_btn.clicked.connect(self.controller.undo)
        self.redo_btn = QPushButton("↪ Redo")
        self.redo_btn.setToolTip("Redo last stroke  [Ctrl+Y]")
        self.redo_btn.clicked.connect(self.controller.redo)
        br.addWidget(self.undo_btn); br.addWidget(self.redo_btn)
        ly.addLayout(br)

        cb = QPushButton("🗑  Clear Canvas"); cb.setObjectName("danger")
        cb.setToolTip("Clear all drawings  [Del]"); cb.clicked.connect(self._clear)
        ly.addWidget(cb)

        sb = QPushButton("💾  Save Drawing"); sb.setObjectName("success")
        sb.setToolTip("Save drawing to PNG  [Ctrl+S]"); sb.clicked.connect(self._save)
        ly.addWidget(sb)

        lb = QPushButton("📂  Load Drawing")
        lb.setToolTip("Load PNG into canvas  [Ctrl+O]"); lb.clicked.connect(self._load)
        ly.addWidget(lb)

        ly.addStretch()
        ly.addWidget(_div())
        ly.addWidget(_sec_lbl("Active Gesture"))
        self.gesture_lbl = QLabel("—")
        self.gesture_lbl.setStyleSheet(
            f"color:{ACCENT_CYAN};font-size:13px;font-weight:bold;"
            f"padding:5px;border:1px solid {BORDER};border-radius:6px;background:{CARD_BG};"
        )
        self.gesture_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ly.addWidget(self.gesture_lbl)
        return p

    def _build_right(self):
        p = QFrame()
        p.setObjectName("panel")
        p.setFixedWidth(224)
        ly = QVBoxLayout(p)
        ly.setContentsMargins(10, 14, 10, 14)
        ly.setSpacing(5)

        ly.addWidget(_glow_lbl("🎭  Filters", ACCENT_MAGENTA))
        ly.addWidget(_div())
        ly.addWidget(_sec_lbl("Built-In Filters"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(294)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        fcon = QWidget()
        fly = QVBoxLayout(fcon)
        fly.setContentsMargins(0, 0, 4, 0); fly.setSpacing(3)
        self._fbtn = {}
        fstyle = (
            f"QPushButton{{background:{CARD_BG};border:1px solid {BORDER};border-radius:6px;"
            f"color:{TEXT_PRIMARY};text-align:left;padding:6px 10px;font-size:12px;}}"
            f"QPushButton:hover{{border-color:{ACCENT_MAGENTA};background:rgba(225,29,150,0.08);}}"
            f"QPushButton:checked{{background:rgba(225,29,150,0.18);border-color:{ACCENT_MAGENTA};"
            f"color:#f472b6;font-weight:bold;}}"
        )
        for ft in FilterType:
            b = QPushButton(FILTER_NAMES[ft]); b.setCheckable(True)
            b.setToolTip(f"Apply '{FILTER_NAMES[ft]}' face filter")
            b.setStyleSheet(fstyle)
            b.clicked.connect(lambda chk, f=ft: self._sel_filter(f))
            self._fbtn[ft] = b; fly.addWidget(b)
        scroll.setWidget(fcon)
        ly.addWidget(scroll)

        ly.addWidget(_sec_lbl("Filter Intensity"))
        self.int_sl = QSlider(Qt.Orientation.Horizontal)
        self.int_sl.setRange(0, 200); self.int_sl.setValue(100)
        self.int_sl.setToolTip("Filter strength (0–200%)")
        self.int_sl.valueChanged.connect(lambda v: self.controller.set_filter_intensity(v / 100.0))
        ly.addWidget(self.int_sl)
        self._il = QLabel("100%"); self._il.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:11px;")
        self.int_sl.valueChanged.connect(lambda v: self._il.setText(f"{v}%"))
        ly.addWidget(self._il)

        ly.addWidget(_div())
        ly.addWidget(_sec_lbl("Custom Filters"))
        self.cust_combo = QComboBox()
        self.cust_combo.setToolTip("Select a saved custom filter")
        self.cust_combo.addItem("— Select Custom Filter —")
        self._refr_cust()
        ly.addWidget(self.cust_combo)

        ar = QHBoxLayout(); ar.setSpacing(5)
        ap = QPushButton("▶ Apply"); ap.setToolTip("Apply the selected custom filter")
        ap.clicked.connect(self._apply_cust)
        dl = QPushButton("🗑"); dl.setFixedWidth(36); dl.setObjectName("danger")
        dl.setToolTip("Delete selected custom filter"); dl.clicked.connect(self._del_cust)
        ar.addWidget(ap); ar.addWidget(dl)
        ly.addLayout(ar)

        ly.addWidget(_div())
        ly.addWidget(_sec_lbl("Drawing → Face Filter"))
        anc_lbl = QLabel("Anchor Point:")
        anc_lbl.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:11px;")
        ly.addWidget(anc_lbl)
        self.anc_combo = QComboBox()
        self.anc_combo.setToolTip("Where to place drawing on face")
        for a in ANCHOR_OPTIONS:
            self.anc_combo.addItem("📍 " + a.replace("_", " ").title(), a)
        ly.addWidget(self.anc_combo)

        cvt = QPushButton("🎨 Convert & Save Filter"); cvt.setObjectName("accent")
        cvt.setToolTip("Save current drawing as a face filter overlay")
        cvt.clicked.connect(self._convert)
        ly.addWidget(cvt)
        ly.addStretch()
        return p

    def _build_bottom(self):
        bar = QFrame(); bar.setObjectName("panel"); bar.setFixedHeight(62)
        bar.setStyleSheet(f"QFrame#panel{{background:{PANEL_BG};border-top:1px solid {BORDER};border-radius:10px;}}")
        ly = QHBoxLayout(bar); ly.setContentsMargins(14, 8, 14, 8); ly.setSpacing(10)

        self.cam_btn = QPushButton("▶  Start Camera"); self.cam_btn.setObjectName("accent")
        self.cam_btn.setFixedWidth(162); self.cam_btn.setToolTip("Start / Stop webcam  [Space]")
        self.cam_btn.clicked.connect(self._toggle_cam)
        ly.addWidget(self.cam_btn)

        self.pause_btn = QPushButton("⏸  Pause"); self.pause_btn.setCheckable(True)
        self.pause_btn.setFixedWidth(100); self.pause_btn.setToolTip("Pause / Resume  [P]")
        self.pause_btn.toggled.connect(lambda v: (
            setattr(self.controller, "is_paused", v),
            self.pause_btn.setText("▶  Resume" if v else "⏸  Pause"),
        ))
        ly.addWidget(self.pause_btn)

        ly.addSpacing(8)
        cl = QLabel("Cam:"); cl.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:11px;")
        ly.addWidget(cl)
        self.cam_combo = QComboBox()
        self.cam_combo.setToolTip("Select camera device (Camera 0, Camera 1...)")
        self.cam_combo.setFixedWidth(115)
        self._populate_cameras()
        self.cam_combo.currentIndexChanged.connect(self._on_cam_selected)
        ly.addWidget(self.cam_combo)


        ly.addStretch()

        self.mode_btn = QPushButton("✏️  Drawing Mode"); self.mode_btn.setCheckable(True)
        self.mode_btn.setFixedWidth(170); self.mode_btn.setToolTip("Switch Drawing ↔ Filter mode  [Tab]")
        self.mode_btn.clicked.connect(self._toggle_mode)
        ly.addWidget(self.mode_btn)

        ly.addStretch()

        self.lm_btn = QPushButton("👁  Landmarks"); self.lm_btn.setCheckable(True)
        self.lm_btn.setFixedWidth(124); self.lm_btn.setToolTip("Toggle landmark overlay  [L]")
        self.lm_btn.toggled.connect(lambda v: setattr(self.controller, "show_landmarks", v))
        ly.addWidget(self.lm_btn)

        snap = QPushButton("📷  Screenshot"); snap.setFixedWidth(132)
        snap.setToolTip("Save screenshot  [Ctrl+Shift+S]"); snap.clicked.connect(self._snap)
        ly.addWidget(snap)
        return bar

    def _build_creations(self):
        pg = QWidget(); ly = QVBoxLayout(pg)
        ly.setContentsMargins(10, 0, 0, 0); ly.setSpacing(12)

        t = QLabel("🖼️  My Creations")
        t.setStyleSheet(f"font-size:26px;font-weight:bold;color:{ACCENT_CYAN};padding:8px 0;")
        ly.addWidget(t)
        s = QLabel("Browse your saved drawings and custom face filters.")
        s.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:13px;padding-bottom:8px;")
        ly.addWidget(s)

        tr = QHBoxLayout(); self._ctg = QButtonGroup(self); self._ctg.setExclusive(True)
        self._cst = QStackedWidget()
        tbs = (
            f"QPushButton{{background:{CARD_BG};border:1px solid {BORDER};"
            f"border-radius:6px;color:{TEXT_SECONDARY};padding:6px 20px;}}"
            f"QPushButton:checked{{background:rgba(6,182,212,0.15);border-color:{ACCENT_CYAN};"
            f"color:{ACCENT_CYAN};font-weight:bold;}}"
            f"QPushButton:hover{{border-color:{ACCENT_CYAN};}}"
        )
        for i, lbl in enumerate(["💾 Saved Drawings", "🎭 Custom Filters"]):
            tb = QPushButton(lbl); tb.setCheckable(True); tb.setStyleSheet(tbs)
            if i == 0: tb.setChecked(True)
            tb.clicked.connect(lambda _, idx=i: self._cst.setCurrentIndex(idx))
            self._ctg.addButton(tb, i); tr.addWidget(tb)
        tr.addStretch()
        ref = QPushButton("🔄 Refresh"); ref.setFixedWidth(100); ref.clicked.connect(self._refr_creations)
        tr.addWidget(ref); ly.addLayout(tr)

        self._dw = QWidget(); self._dg = QGridLayout(self._dw)
        self._dg.setSpacing(10); self._dg.setAlignment(Qt.AlignmentFlag.AlignTop)
        ds = QScrollArea(); ds.setWidgetResizable(True); ds.setWidget(self._dw)
        self._cst.addWidget(ds)

        self._fw = QWidget(); self._fg = QGridLayout(self._fw)
        self._fg.setSpacing(10); self._fg.setAlignment(Qt.AlignmentFlag.AlignTop)
        fs = QScrollArea(); fs.setWidgetResizable(True); fs.setWidget(self._fw)
        self._cst.addWidget(fs)

        ly.addWidget(self._cst, 1)
        self._refr_creations()
        return pg

    def _build_settings(self):
        pg = QWidget(); ly = QVBoxLayout(pg)
        ly.setContentsMargins(10, 0, 0, 0); ly.setSpacing(16)
        t = QLabel("⚙️  Settings")
        t.setStyleSheet(f"font-size:26px;font-weight:bold;color:{ACCENT_PURPLE};padding:8px 0;")
        ly.addWidget(t)

        scr = QScrollArea(); scr.setWidgetResizable(True)
        con = QWidget(); sl = QVBoxLayout(con); sl.setSpacing(16); sl.setContentsMargins(0, 0, 16, 0)

        # Camera
        cg = QGroupBox("Camera"); cf = QFormLayout(cg)
        cf.setSpacing(8); cf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._sc_idx = QSpinBox(); self._sc_idx.setRange(0, 9); self._sc_idx.setValue(self.controller.camera.device_index)
        cf.addRow("Device Index:", self._sc_idx)
        self._sc_w = QSpinBox(); self._sc_w.setRange(320, 1920); self._sc_w.setSingleStep(160); self._sc_w.setValue(self.controller.camera.target_width)
        cf.addRow("Width:", self._sc_w)
        self._sc_h = QSpinBox(); self._sc_h.setRange(240, 1080); self._sc_h.setSingleStep(120); self._sc_h.setValue(self.controller.camera.target_height)
        cf.addRow("Height:", self._sc_h)
        self._sc_fps = QSpinBox(); self._sc_fps.setRange(10, 60); self._sc_fps.setValue(self.controller.camera.target_fps)
        cf.addRow("Target FPS:", self._sc_fps)
        sl.addWidget(cg)

        # Tracking
        tg = QGroupBox("Hand & Face Tracking"); tf = QFormLayout(tg)
        tf.setSpacing(8); tf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._sh = QDoubleSpinBox(); self._sh.setRange(0.1, 1.0); self._sh.setSingleStep(0.05); self._sh.setValue(0.7)
        tf.addRow("Hand Detection Confidence:", self._sh)
        self._sht = QDoubleSpinBox(); self._sht.setRange(0.1, 1.0); self._sht.setSingleStep(0.05); self._sht.setValue(0.5)
        tf.addRow("Hand Tracking Confidence:", self._sht)
        self._sf = QDoubleSpinBox(); self._sf.setRange(0.1, 1.0); self._sf.setSingleStep(0.05); self._sf.setValue(0.5)
        tf.addRow("Face Detection Confidence:", self._sf)
        self._ss = QDoubleSpinBox(); self._ss.setRange(0.0, 1.0); self._ss.setSingleStep(0.05); self._ss.setValue(0.7)
        tf.addRow("Smoothing Factor:", self._ss)
        sl.addWidget(tg)

        # Performance
        pg2 = QGroupBox("Performance"); pf = QFormLayout(pg2)
        pf.setSpacing(8); pf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._sp = QSpinBox(); self._sp.setRange(10, 1000); self._sp.setValue(self.controller.drawing_engine.max_particles)
        pf.addRow("Max Particles:", self._sp)
        sl.addWidget(pg2)

        # Display
        dg = QGroupBox("Display"); dl = QVBoxLayout(dg); dl.setSpacing(8)
        self._sfps = QCheckBox("Show FPS on video"); self._sfps.setChecked(self.controller.show_fps)
        self._slm = QCheckBox("Show landmark overlay"); self._slm.setChecked(self.controller.show_landmarks)
        self._sgst = QCheckBox("Show gesture indicator on video"); self._sgst.setChecked(self.controller.show_gesture)
        for w in [self._sfps, self._slm, self._sgst]: dl.addWidget(w)
        sl.addWidget(dg)

        # Gesture
        gg = QGroupBox("Gesture Recognition"); gf = QFormLayout(gg)
        gf.setSpacing(8); gf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._sdeb = QSpinBox(); self._sdeb.setRange(1, 30); self._sdeb.setValue(self.controller.gesture_rec.debounce_frames)
        gf.addRow("Debounce Frames:", self._sdeb)
        self._sgconf = QDoubleSpinBox(); self._sgconf.setRange(0.5, 1.0); self._sgconf.setSingleStep(0.05); self._sgconf.setValue(self.controller.gesture_rec.confidence_threshold)
        gf.addRow("Confidence Threshold:", self._sgconf)
        sl.addWidget(gg)

        # Drawing & Assisted Tools
        drg = QGroupBox("Drawing & Assisted Tools"); drf = QVBoxLayout(drg); drf.setSpacing(8)
        self._ssnap = QCheckBox("Auto-snap geometric shapes (circles, lines, rectangles)")
        self._ssnap.setChecked(self.controller.auto_snap_shapes)
        drf.addWidget(self._ssnap)
        sl.addWidget(drg)

        sl.addStretch()
        scr.setWidget(con); ly.addWidget(scr, 1)

        row = QHBoxLayout()
        ab = QPushButton("💾  Apply Settings"); ab.setObjectName("accent"); ab.setFixedWidth(180)
        ab.clicked.connect(self._apply_settings)
        row.addStretch(); row.addWidget(ab); ly.addLayout(row)
        return pg

    # ── Shortcuts ──────────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self, self.controller.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.controller.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.controller.redo)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save)
        QShortcut(QKeySequence("Ctrl+O"), self, self._load)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, self._snap)
        QShortcut(QKeySequence("Delete"), self, self._clear)
        QShortcut(QKeySequence("Space"), self, self._toggle_cam)
        QShortcut(QKeySequence("P"), self, lambda: self.pause_btn.toggle())
        QShortcut(QKeySequence("Tab"), self, self._toggle_mode)
        QShortcut(QKeySequence("E"), self, lambda: self.eraser_btn.toggle())
        QShortcut(QKeySequence("L"), self, lambda: self.lm_btn.toggle())
        QShortcut(QKeySequence("F1"), self, self._show_shortcuts_dialog)
        QShortcut(QKeySequence("?"), self, self._show_shortcuts_dialog)
        QShortcut(QKeySequence("["), self, lambda: self.size_sl.setValue(max(1, self.size_sl.value() - 2)))
        QShortcut(QKeySequence("]"), self, lambda: self.size_sl.setValue(min(50, self.size_sl.value() + 2)))
        QShortcut(QKeySequence("1"), self, lambda: (self._nbtns[0].setChecked(True), self._nav(0)))
        QShortcut(QKeySequence("2"), self, lambda: (self._nbtns[1].setChecked(True), self._nav(1)))
        QShortcut(QKeySequence("3"), self, lambda: (self._nbtns[2].setChecked(True), self._nav(2)))
        QShortcut(QKeySequence("4"), self, lambda: (self._nbtns[3].setChecked(True), self._nav(3)))


    # ── Nav ────────────────────────────────────────────────────────────────

    def _nav(self, idx):
        if idx in (0, 1):
            self._cs.setCurrentIndex(0)
            self._lp.setVisible(True); self._rp.setVisible(True)
            if idx == 0:
                self.controller.mode = AppMode.DRAWING
                self.mode_btn.setChecked(False); self.mode_btn.setText("✏️  Drawing Mode")
            else:
                self.controller.mode = AppMode.FACE_FILTER
                self.mode_btn.setChecked(True); self.mode_btn.setText("🎭  Filter Mode")
        elif idx == 2:
            self._cs.setCurrentIndex(1)
            self._lp.setVisible(False); self._rp.setVisible(False)
            self._refr_creations()
        elif idx == 3:
            self._cs.setCurrentIndex(2)
            self._lp.setVisible(False); self._rp.setVisible(False)

    # ── Frame loop ─────────────────────────────────────────────────────────

    def _on_frame_ready(self, frame, hand_data, face_data):
        with self._frame_lock:
            self._pending_frame = frame
        self.frame_received.emit(frame)

    def _display_frame(self, frame):
        self.video_label.set_frame(frame)

    def _update_ui(self):
        if not self.controller.camera.is_running:
            self._fps_lbl.setText("FPS —"); self._dot.set_off(); return
        self._dot.set_ok()
        fps = self.controller.camera.fps_actual
        self._fps_lbl.setText(f"FPS {fps:.0f}")
        lbl = self.controller.gesture_label or "—"
        self.gesture_lbl.setText(lbl)
        if self.controller.mode == AppMode.FACE_FILTER:
            if not self.mode_btn.isChecked():
                self.mode_btn.setChecked(True); self.mode_btn.setText("🎭  Filter Mode")
        else:
            if self.mode_btn.isChecked():
                self.mode_btn.setChecked(False); self.mode_btn.setText("✏️  Drawing Mode")
        mode = "Drawing" if self.controller.mode == AppMode.DRAWING else "Face Filter"
        h = "✓" if self.controller.hand_data else "—"
        f = "✓" if self.controller.face_data else "—"
        self._sb.showMessage(
            f"  FPS: {fps:.1f}  │  Mode: {mode}  │  Hand: {h}  │  Face: {f}  │  Gesture: {lbl}"
        )

    # ── Actions ────────────────────────────────────────────────────────────

    def _auto_start_camera(self):
        if not self.controller.camera.is_running:
            self._toggle_cam()

    def _populate_cameras(self):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        cams = self.controller.camera.list_cameras()
        if not cams:
            cams = [0]
        for c in cams:
            label = "📷 Default Cam (0)" if c == 0 else f"📷 Cam {c}"
            self.cam_combo.addItem(label, c)
        idx = self.cam_combo.findData(self.controller.camera.device_index)
        if idx >= 0:
            self.cam_combo.setCurrentIndex(idx)
        self.cam_combo.blockSignals(False)

    def _on_cam_selected(self, idx):
        dev = self.cam_combo.itemData(idx)
        if dev is not None and dev != self.controller.camera.device_index:
            was_running = self.controller.camera.is_running
            if was_running:
                self.controller.stop_camera()
            self.controller.camera.device_index = dev
            if was_running:
                if self.controller.start_camera(dev):
                    self.cam_btn.setText("⏹  Stop Camera")
                    self._sb.showMessage(f"Camera {dev} started")
                    self._dot.set_ok()
                else:
                    self._dot.set_warn()
                    self._err(f"Could not open Camera {dev}.")

    def _toggle_cam(self):
        if not self.controller.camera.is_running:
            dev = self.cam_combo.currentData() if hasattr(self, "cam_combo") else self.controller.camera.device_index
            if dev is None: dev = 0
            if self.controller.start_camera(dev):
                self.cam_btn.setText("⏹  Stop Camera")
                actual_idx = self.controller.camera.device_index
                self._sb.showMessage(f"Camera {actual_idx} active — ready to draw")
                self._dot.set_ok()
                if hasattr(self, "cam_combo"):
                    idx = self.cam_combo.findData(actual_idx)
                    if idx >= 0 and idx != self.cam_combo.currentIndex():
                        self.cam_combo.blockSignals(True)
                        self.cam_combo.setCurrentIndex(idx)
                        self.cam_combo.blockSignals(False)
            else:
                self._dot.set_warn()
                self._err(
                    "Could not open camera.\n\n"
                    "• Check no other app is using it (Zoom, Teams…)\n"
                    "• Try selecting another camera in the 'Cam' dropdown\n"
                    "• Check your Windows privacy settings (Settings > Privacy > Camera)"
                )
        else:
            self.controller.stop_camera()
            self.cam_btn.setText("▶  Start Camera")
            self.video_label.setText("📷  Camera stopped\n\nClick  ▶ Start Camera  to resume")
            self._sb.showMessage("Camera stopped"); self._dot.set_off(); self._fps_lbl.setText("FPS —")


    def _toggle_mode(self):
        if self.controller.mode == AppMode.DRAWING:
            self.controller.mode = AppMode.FACE_FILTER
            self.mode_btn.setChecked(True); self.mode_btn.setText("🎭  Filter Mode")
            self._nbtns[1].setChecked(True)
        else:
            self.controller.mode = AppMode.DRAWING
            self.mode_btn.setChecked(False); self.mode_btn.setText("✏️  Drawing Mode")
            self._nbtns[0].setChecked(True)

    def _on_brush_change(self, idx):
        brush = self.brush_combo.itemData(idx)
        if brush:
            self.controller.set_brush(brush)
            if self.eraser_btn.isChecked(): self.eraser_btn.setChecked(False)

    def _on_eraser(self, active):
        if active: self.controller.set_brush(BrushType.ERASER)
        else: self._on_brush_change(self.brush_combo.currentIndex())

    def _clear(self):
        if not self.controller.drawing_engine.strokes and self.controller.drawing_engine.canvas.max() == 0:
            return
        if QMessageBox.question(
            self, "Clear Canvas", "Clear the entire drawing?\n(Undoable with Ctrl+Z)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self.controller.clear_canvas()
            self._sb.showMessage("Canvas cleared")

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Drawing", self.controller._saved_drawings_dir, "PNG Image (*.png);;All Files (*)"
        )
        if path:
            if not path.lower().endswith(".png"): path += ".png"
            if self.controller.drawing_engine.save_drawing(path):
                self._sb.showMessage(f"💾 Saved: {os.path.basename(path)}")
            else:
                self._err("Failed to save drawing.")

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Drawing", self.controller._saved_drawings_dir, "PNG Image (*.png);;All Files (*)"
        )
        if path:
            if self.controller.drawing_engine.load_drawing(path):
                self._sb.showMessage(f"📂 Loaded: {os.path.basename(path)}")
            else:
                self._err("Failed to load drawing. Ensure it is a valid PNG.")

    def _sel_filter(self, ft):
        for f, b in self._fbtn.items(): b.setChecked(f == ft)
        self.controller.set_face_filter(ft)
        self.controller.mode = AppMode.DRAWING if ft == FilterType.NONE else AppMode.FACE_FILTER
        self._sb.showMessage(f"Filter: {FILTER_NAMES[ft]}")

    def _apply_cust(self):
        name = self.cust_combo.currentText()
        if name.startswith("—"): self._sb.showMessage("Select a custom filter first"); return
        self.controller.apply_custom_filter(name)
        self.controller.mode = AppMode.FACE_FILTER
        self._sb.showMessage(f"Applied: {name}")

    def _del_cust(self):
        name = self.cust_combo.currentText()
        if name.startswith("—"): return
        if QMessageBox.question(
            self, "Delete Filter", f"Delete '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self.controller.custom_filter_manager.delete_filter(name)
            self._refr_cust()
            self._sb.showMessage(f"Deleted: {name}")

    def _convert(self):
        name, ok = QInputDialog.getText(
            self, "Save Custom Filter", "Name for your custom filter:",
            QLineEdit.EchoMode.Normal, ""
        )
        if not ok or not name.strip(): return
        anchor = self.anc_combo.itemData(self.anc_combo.currentIndex()) or "forehead"
        if self.controller.convert_drawing_to_filter(name=name.strip(), anchor=anchor):
            self._refr_cust()
            QMessageBox.information(self, "Filter Saved ✅",
                f"Custom filter '{name}' saved!\n\nSelect it from the dropdown and click '▶ Apply'.")
        else:
            QMessageBox.warning(self, "Save Failed",
                "No drawing on canvas.\nDraw something first, then convert.")

    def _refr_cust(self):
        self.cust_combo.clear()
        self.cust_combo.addItem("— Select Custom Filter —")
        for cf in self.controller.custom_filter_manager.list_filters():
            self.cust_combo.addItem(cf.name)

    def _snap(self):
        path = self.controller.take_snapshot()
        if path: self._sb.showMessage(f"📷 Snapshot: {os.path.basename(path)}")
        else: self._sb.showMessage("Cannot take snapshot — camera not active")

    def _on_cam_error(self, msg): self._dot.set_warn(); self._sb.showMessage(f"⚠️ {msg}")
    def _on_status(self, msg): self._sb.showMessage(msg)
    def _err(self, msg): QMessageBox.critical(self, "DoodleLens — Error", msg)

    def _apply_settings(self):
        self.controller.camera.device_index = self._sc_idx.value()
        self.controller.camera.target_width = self._sc_w.value()
        self.controller.camera.target_height = self._sc_h.value()
        self.controller.camera.target_fps = self._sc_fps.value()
        self.controller.drawing_engine.max_particles = self._sp.value()
        self.controller.show_fps = self._sfps.isChecked()
        self.controller.show_landmarks = self._slm.isChecked()
        self.controller.show_gesture = self._sgst.isChecked()
        self.controller.gesture_rec.debounce_frames = self._sdeb.value()
        self.controller.gesture_rec.confidence_threshold = self._sgconf.value()
        self.controller.auto_snap_shapes = self._ssnap.isChecked()
        self.snap_btn.setChecked(self._ssnap.isChecked())
        self.controller.hand_tracker.smoothing_factor = self._ss.value()
        self.controller.face_tracker.smoothing_factor = self._ss.value()
        if hasattr(self, "cam_combo"):
            idx = self.cam_combo.findData(self._sc_idx.value())
            if idx >= 0:
                self.cam_combo.setCurrentIndex(idx)
        self.lm_btn.setChecked(self._slm.isChecked())

        self.controller.save_settings()
        QMessageBox.information(self, "Settings Applied",
            "Settings applied and saved successfully.\nRestart the camera for new resolution / device to take effect.")
        self._sb.showMessage("Settings applied and saved")

    # ── My Creations ───────────────────────────────────────────────────────

    def _refr_creations(self):
        self._pop_dwg(); self._pop_flt()

    def _pop_dwg(self):
        while self._dg.count():
            item = self._dg.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        d = self.controller._saved_drawings_dir
        files = sorted(
            [f for f in os.listdir(d) if f.lower().endswith(".png")], reverse=True
        ) if os.path.isdir(d) else []
        if not files:
            lbl = QLabel("No saved drawings yet.\n\nUse 💾 Save Drawing to save your work!")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:14px;padding:40px;")
            self._dg.addWidget(lbl, 0, 0); return
        for i, f in enumerate(files):
            self._dg.addWidget(self._card(os.path.join(d, f), f, False), i // 4, i % 4)

    def _pop_flt(self):
        while self._fg.count():
            item = self._fg.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        fls = self.controller.custom_filter_manager.list_filters()
        if not fls:
            lbl = QLabel("No custom filters yet.\n\nUse 🎨 Convert & Save Filter to create one!")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:14px;padding:40px;")
            self._fg.addWidget(lbl, 0, 0); return
        for i, cf in enumerate(fls):
            self._fg.addWidget(self._card(cf.image_path, cf.name, True), i // 4, i % 4)

    def _card(self, fpath, name, is_filter):
        card = QFrame(); card.setFixedSize(192, 222)
        card.setStyleSheet(
            f"QFrame{{background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;}}"
            f"QFrame:hover{{border-color:{ACCENT_CYAN};background:{CARD_HOVER};}}"
        )
        ly = QVBoxLayout(card); ly.setContentsMargins(8, 8, 8, 8); ly.setSpacing(6)
        thumb = QLabel(); thumb.setFixedSize(176, 152)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet(f"background:#08080f;border-radius:6px;border:1px solid {BORDER};")
        if os.path.exists(fpath):
            try:
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                    elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    h, w = img.shape[:2]
                    bg = np.full((h, w, 4), (8, 8, 15, 255), dtype=np.uint8)
                    a = img[:, :, 3:4].astype(float) / 255.0
                    for c in range(3):
                        bg[:, :, c] = (bg[:, :, c] * (1 - a[:, :, 0]) + img[:, :, c] * a[:, :, 0]).astype(np.uint8)
                    prev = cv2.cvtColor(bg, cv2.COLOR_BGRA2RGB)
                    h2, w2 = prev.shape[:2]
                    qi = QImage(prev.data, w2, h2, w2 * 3, QImage.Format.Format_RGB888)
                    thumb.setPixmap(QPixmap.fromImage(qi).scaled(
                        176, 152, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                else: thumb.setText("⚠️ Bad image")
            except Exception: thumb.setText("⚠️ Error")
        else: thumb.setText("📁 Missing")
        ly.addWidget(thumb)
        short = name[:22] + "…" if len(name) > 22 else name
        nl = QLabel(short); nl.setStyleSheet(f"color:{TEXT_PRIMARY};font-size:11px;font-weight:bold;")
        nl.setAlignment(Qt.AlignmentFlag.AlignCenter); nl.setToolTip(name)
        ly.addWidget(nl)

        act_row = QHBoxLayout(); act_row.setSpacing(4)
        if is_filter:
            ab = QPushButton("▶ Apply")
            ab.setStyleSheet(
                f"QPushButton{{background:rgba(225,29,150,0.15);border:1px solid {ACCENT_MAGENTA};"
                f"color:#f472b6;border-radius:5px;padding:4px;font-size:11px;}}"
                f"QPushButton:hover{{background:rgba(225,29,150,0.30);}}"
            )
            ab.clicked.connect(lambda _, n=name: self._apply_named(n))
            db = QPushButton("🗑")
            db.setFixedWidth(28); db.setObjectName("danger")
            db.setToolTip(f"Delete filter '{name}'")
            db.clicked.connect(lambda _, n=name: self._del_named_filter(n))
            act_row.addWidget(ab, 1); act_row.addWidget(db)
        else:
            ab = QPushButton("📂 Load")
            ab.setStyleSheet(
                f"QPushButton{{background:rgba(6,182,212,0.10);border:1px solid {ACCENT_CYAN};"
                f"color:{ACCENT_CYAN};border-radius:5px;padding:4px;font-size:11px;}}"
                f"QPushButton:hover{{background:rgba(6,182,212,0.25);}}"
            )
            ab.clicked.connect(lambda _, p=fpath: self._load_named(p))
            db = QPushButton("🗑")
            db.setFixedWidth(28); db.setObjectName("danger")
            db.setToolTip(f"Delete drawing '{name}'")
            db.clicked.connect(lambda _, p=fpath: self._del_named_drawing(p))
            act_row.addWidget(ab, 1); act_row.addWidget(db)
        ly.addLayout(act_row)
        return card

    def _del_named_drawing(self, path):
        fname = os.path.basename(path)
        if QMessageBox.question(
            self, "Delete Drawing", f"Delete '{fname}'?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            try:
                if os.path.exists(path):
                    os.remove(path)
                self._refr_creations()
                self._sb.showMessage(f"Deleted drawing: {fname}")
            except Exception as e:
                self._err(f"Could not delete {fname}: {e}")

    def _del_named_filter(self, name):
        if QMessageBox.question(
            self, "Delete Filter", f"Delete custom filter '{name}'?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self.controller.custom_filter_manager.delete_filter(name)
            self._refr_cust()
            self._refr_creations()
            self._sb.showMessage(f"Deleted custom filter: {name}")

    def _show_shortcuts_dialog(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("DoodleLens — Shortcuts & Gestures Guide")
        dlg.setTextFormat(Qt.TextFormat.RichText)
        info = f"""
        <h3 style='color:{ACCENT_PURPLE}; margin-bottom:8px;'>✨ Gestures Guide</h3>
        <table border='0' cellpadding='4' style='color:#f0f0ff; font-size:12px;'>
          <tr><td><b>☝️ Index Extended:</b></td><td>Draw with active brush & color</td></tr>
          <tr><td><b>✋ Open Palm:</b></td><td>Pause drawing / move cursor freely</td></tr>
          <tr><td><b>✊ Closed Fist:</b></td><td>Clear canvas</td></tr>
          <tr><td><b>👍 Thumbs Up:</b></td><td>Take screenshot</td></tr>
          <tr><td><b>✌️ Two Fingers:</b></td><td>Toggle Drawing ↔ Face Filter mode</td></tr>
        </table>
        <br>
        <h3 style='color:{ACCENT_CYAN}; margin-bottom:8px;'>⌨️ Keyboard Shortcuts</h3>
        <table border='0' cellpadding='4' style='color:#f0f0ff; font-size:12px;'>
          <tr><td><b>Space</b></td><td>Start / Stop Camera</td></tr>
          <tr><td><b>P</b></td><td>Pause / Resume Camera</td></tr>
          <tr><td><b>Tab</b></td><td>Switch Drawing / Filter Mode</td></tr>
          <tr><td><b>Ctrl + Z / Ctrl + Y</b></td><td>Undo / Redo Stroke</td></tr>
          <tr><td><b>Ctrl + S / Ctrl + O</b></td><td>Save / Load Drawing</td></tr>
          <tr><td><b>Ctrl + Shift + S</b></td><td>Take Screenshot</td></tr>
          <tr><td><b>E</b></td><td>Toggle Eraser</td></tr>
          <tr><td><b>L</b></td><td>Toggle Landmarks Overlay</td></tr>
          <tr><td><b>[ / ]</b></td><td>Decrease / Increase Brush Size</td></tr>
          <tr><td><b>Del</b></td><td>Clear Canvas</td></tr>
          <tr><td><b>1 / 2 / 3 / 4</b></td><td>Switch Views (Draw / Filter / Creations / Settings)</td></tr>
          <tr><td><b>F1 or ?</b></td><td>Open this Shortcuts Guide</td></tr>
        </table>
        """
        dlg.setText(info)
        dlg.exec()

    def _apply_named(self, name):
        self.controller.apply_custom_filter(name)
        self.controller.mode = AppMode.FACE_FILTER
        self._nbtns[1].setChecked(True); self._nav(1)
        self._sb.showMessage(f"Applied: {name}")

    def _load_named(self, path):
        if self.controller.drawing_engine.load_drawing(path):
            self._nbtns[0].setChecked(True); self._nav(0)
            self._sb.showMessage(f"Loaded: {os.path.basename(path)}")
        else:
            self._err(f"Failed to load:\n{path}")


    # ── Window Events ──────────────────────────────────────────────────────
    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_camera_started_once", False):
            self._camera_started_once = True
            QTimer.singleShot(250, self._auto_start_camera)

    def closeEvent(self, event):
        self._timer.stop()
        try: self.controller.shutdown()
        except Exception: pass
        event.accept()

