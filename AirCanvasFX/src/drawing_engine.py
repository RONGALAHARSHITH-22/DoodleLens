"""
Drawing Engine - DoodleLens
Manages the drawing canvas, brush effects, undo/redo, and rendering.
"""

import cv2
import numpy as np
import math
import random
import time
from collections import deque
from typing import List, Tuple, Optional, Dict
from enum import Enum, auto


class BrushType(Enum):
    NEON_PEN = "neon_pen"
    RAINBOW = "rainbow"
    LASER_TRAIL = "laser_trail"
    GLOWING_MARKER = "glowing_marker"
    PARTICLE = "particle"
    SPARKLE = "sparkle"
    FIRE = "fire"
    ELECTRIC = "electric"
    PASTEL = "pastel"
    RAINBOW_SPIRAL = "rainbow_spiral"
    ERASER = "eraser"


BRUSH_NAMES = {
    BrushType.NEON_PEN: "Neon Pen",
    BrushType.RAINBOW: "Rainbow Brush",
    BrushType.LASER_TRAIL: "Laser Trail",
    BrushType.GLOWING_MARKER: "Glowing Marker",
    BrushType.PARTICLE: "Particle Brush",
    BrushType.SPARKLE: "Sparkle Trail",
    BrushType.FIRE: "Fire Trail",
    BrushType.ELECTRIC: "Electric Lightning",
    BrushType.PASTEL: "Soft Pastel",
    BrushType.RAINBOW_SPIRAL: "Rainbow Spiral",
    BrushType.ERASER: "Eraser",
}


class Particle:
    """Represents a single particle in particle-based effects."""
    __slots__ = ["x", "y", "vx", "vy", "life", "max_life", "color", "size"]

    def __init__(self, x: float, y: float, color: Tuple, max_life: int = 20):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = max_life
        self.max_life = max_life
        self.color = color
        self.size = random.randint(2, 5)

    def update(self) -> bool:
        """Update position, return True if alive."""
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # gravity
        self.life -= 1
        return self.life > 0


class Stroke:
    """A single drawing stroke with all its points and style info."""

    def __init__(self, brush: BrushType, color: Tuple, size: int,
                 glow: int, opacity: float):
        self.brush = brush
        self.color = color
        self.size = size
        self.glow = glow
        self.opacity = opacity
        self.points: List[Tuple[int, int]] = []
        self.timestamp = time.time()

    def add_point(self, pt: Tuple[int, int]):
        self.points.append(pt)


class DrawingEngine:
    """
    Manages the drawing canvas, multiple strokes, particle effects,
    undo/redo stack, and renders each brush type.
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 max_particles: int = 300):
        self.width = width
        self.height = height
        self.max_particles = max_particles

        # Canvas layers
        self.canvas = np.zeros((height, width, 4), dtype=np.uint8)  # BGRA
        self._stroke_buffer: Optional[np.ndarray] = None

        # Strokes
        self.strokes: List[Stroke] = []
        self.current_stroke: Optional[Stroke] = None
        self.undo_stack: List[List[Stroke]] = []
        self.redo_stack: List[List[Stroke]] = []

        # Particles
        self.particles: List[Particle] = []

        # Settings
        self.brush_type = BrushType.NEON_PEN
        self.brush_color = (0, 255, 255)  # Cyan
        self.brush_size = 8
        self.glow_intensity = 5
        self.opacity = 1.0
        self.trail_length = 30

        # State
        self.is_drawing = False
        self._rainbow_hue = 0.0
        self._spiral_angle = 0.0
        self._frame_count = 0
        self._last_point: Optional[Tuple[int, int]] = None

    def start_stroke(self):
        """Begin a new stroke."""
        if self.current_stroke is not None:
            self._finalize_stroke()
        self.current_stroke = Stroke(
            self.brush_type, self.brush_color,
            self.brush_size, self.glow_intensity, self.opacity
        )
        self.is_drawing = True
        self._last_point = None

    def add_point(self, pt: Tuple[int, int]):
        """Add a point to the current stroke and draw it."""
        if self.current_stroke is None:
            self.start_stroke()

        x, y = int(pt[0]), int(pt[1])
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return

        self.current_stroke.add_point((x, y))

        # Draw onto canvas
        if self._last_point is not None:
            self._draw_segment(self.canvas, self._last_point, (x, y),
                               self.current_stroke)
        else:
            self._draw_dot(self.canvas, (x, y), self.current_stroke)

        self._last_point = (x, y)

        # Spawn particles
        if self.brush_type in (BrushType.PARTICLE, BrushType.SPARKLE,
                                BrushType.FIRE):
            self._spawn_particles((x, y))

    def end_stroke(self):
        """Finalize and commit the current stroke."""
        self._finalize_stroke()
        self.is_drawing = False
        self._last_point = None

    def _finalize_stroke(self):
        if self.current_stroke and len(self.current_stroke.points) > 0:
            self.undo_stack.append(list(self.strokes))
            self.strokes.append(self.current_stroke)
            self.redo_stack.clear()
        self.current_stroke = None

    def update_particles(self):
        """Update all active particles."""
        self.particles = [p for p in self.particles if p.update()]
        self._frame_count += 1
        self._rainbow_hue = (self._rainbow_hue + 2) % 360

    def _spawn_particles(self, pt: Tuple[int, int]):
        """Spawn particles at a given point."""
        if len(self.particles) >= self.max_particles:
            return
        n = min(5, self.max_particles - len(self.particles))
        for _ in range(n):
            color = self._get_effect_color()
            self.particles.append(Particle(pt[0], pt[1], color))

    def _get_effect_color(self) -> Tuple:
        """Get color for effects based on brush type."""
        if self.brush_type == BrushType.RAINBOW:
            return self._hsv_to_bgr(self._rainbow_hue, 1.0, 1.0)
        elif self.brush_type == BrushType.FIRE:
            h = random.choice([0, 15, 30])
            return self._hsv_to_bgr(h, 1.0, 1.0)
        elif self.brush_type == BrushType.ELECTRIC:
            return random.choice([(255, 255, 0), (200, 200, 255), (255, 255, 255)])
        return self.brush_color

    @staticmethod
    def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to BGR."""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))

    def _draw_segment(self, canvas: np.ndarray, p1: Tuple, p2: Tuple,
                       stroke: Stroke):
        """Draw a segment between two points using the stroke brush."""
        dist = max(1, int(math.hypot(p2[0] - p1[0], p2[1] - p1[1])))
        # Interpolate intermediate points for smooth lines
        steps = max(1, dist // 2)
        for i in range(steps + 1):
            t = i / steps
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            self._draw_dot(canvas, (x, y), stroke)

    def _draw_dot(self, canvas: np.ndarray, pt: Tuple, stroke: Stroke):
        """Draw a single brush dot at pt."""
        x, y = pt
        b = stroke.brush
        color = list(stroke.color)
        size = stroke.size
        glow = stroke.glow

        if b == BrushType.ERASER:
            cv2.circle(canvas, (x, y), size * 3, (0, 0, 0, 0), -1)
            return

        if b == BrushType.RAINBOW:
            color = list(self._hsv_to_bgr(self._rainbow_hue, 1.0, 1.0))
        elif b == BrushType.RAINBOW_SPIRAL:
            hue = (self._rainbow_hue + self._spiral_angle * 5) % 360
            color = list(self._hsv_to_bgr(hue, 1.0, 1.0))
            self._spiral_angle += 5

        alpha = int(stroke.opacity * 255)

        if b in (BrushType.NEON_PEN, BrushType.GLOWING_MARKER,
                 BrushType.RAINBOW, BrushType.RAINBOW_SPIRAL):
            # Glow layers (additive)
            for layer in range(glow, 0, -1):
                layer_alpha = int(alpha * layer / (glow * 2))
                r = size + layer * 3
                cv2.circle(canvas, (x, y), r,
                           (*color, layer_alpha), -1)
            cv2.circle(canvas, (x, y), size, (*color, alpha), -1)

        elif b == BrushType.LASER_TRAIL:
            # Bright core with wide glow
            for r_mult in [4, 2, 1]:
                a = int(alpha / (r_mult * 1.5))
                cv2.circle(canvas, (x, y), size * r_mult, (*color, a), -1)
            cv2.circle(canvas, (x, y), max(1, size // 2),
                       (255, 255, 255, alpha), -1)

        elif b == BrushType.PASTEL:
            # Soft transparent strokes
            a = int(alpha * 0.5)
            cv2.circle(canvas, (x, y), size * 2, (*color, a), -1)
            cv2.circle(canvas, (x, y), size, (*color, int(alpha * 0.8)), -1)

        elif b == BrushType.ELECTRIC:
            # Jittery electric lines
            color = list(self._get_effect_color())
            if random.random() > 0.3:
                jx = x + random.randint(-5, 5)
                jy = y + random.randint(-5, 5)
                cv2.line(canvas, (x, y), (jx, jy), (*color, alpha), 2)
            cv2.circle(canvas, (x, y), max(1, size // 2), (255, 255, 255, alpha), -1)

        elif b in (BrushType.PARTICLE, BrushType.SPARKLE, BrushType.FIRE):
            # Particles are rendered separately; draw small core
            cv2.circle(canvas, (x, y), max(1, size // 2), (*color, alpha), -1)

        else:
            cv2.circle(canvas, (x, y), size, (*color, alpha), -1)

    def render_particles(self, canvas: np.ndarray):
        """Draw all live particles onto the canvas."""
        for p in self.particles:
            ratio = p.life / p.max_life
            alpha = int(255 * ratio)
            r = max(1, int(p.size * ratio))
            px, py = int(p.x), int(p.y)
            if 0 <= px < self.width and 0 <= py < self.height:
                color = list(p.color)
                if self.brush_type == BrushType.SPARKLE:
                    # Star shape
                    for dx, dy in [(r, 0), (-r, 0), (0, r), (0, -r)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            cv2.line(canvas, (px, py), (nx, ny),
                                     (*color, alpha), 1)
                else:
                    cv2.circle(canvas, (px, py), r, (*color, alpha), -1)

    def get_composite_frame(self, camera_frame: np.ndarray) -> np.ndarray:
        """Blend drawing canvas + particles onto camera frame (ultra-fast vectorized)."""
        # Render particles on canvas copy only if there are live particles
        if self.particles:
            canvas_with_particles = self.canvas.copy()
            self.render_particles(canvas_with_particles)
        else:
            canvas_with_particles = self.canvas

        # Quick check: if canvas has no visible strokes, return camera_frame immediately
        alpha = canvas_with_particles[:, :, 3]
        if not np.any(alpha > 0):
            return camera_frame

        # Apply Gaussian blur for glow to certain brushes
        if self.brush_type in (BrushType.NEON_PEN, BrushType.GLOWING_MARKER,
                                BrushType.LASER_TRAIL, BrushType.RAINBOW):
            blurred = cv2.GaussianBlur(canvas_with_particles, (0, 0), 3)
            canvas_with_particles = cv2.addWeighted(
                canvas_with_particles, 0.7, blurred, 0.5, 0)
            alpha = canvas_with_particles[:, :, 3]

        out = camera_frame.copy()

        # Resize canvas if needed
        if canvas_with_particles.shape[:2] != out.shape[:2]:
            canvas_with_particles = cv2.resize(
                canvas_with_particles,
                (out.shape[1], out.shape[0])
            )
            alpha = canvas_with_particles[:, :, 3]

        # Fast vectorized integer alpha blending (under 1ms)
        mask = alpha > 0
        if np.any(mask):
            a = alpha[mask].astype(np.uint16)
            inv_a = (255 - a).astype(np.uint16)
            for c in range(3):
                fg = canvas_with_particles[:, :, c][mask].astype(np.uint16)
                bg = out[:, :, c][mask].astype(np.uint16)
                out[:, :, c][mask] = ((fg * a + bg * inv_a) // 255).astype(np.uint8)

        return out


    def get_canvas_image(self) -> np.ndarray:
        """Return BGRA canvas for saving."""
        c = self.canvas.copy()
        self.render_particles(c)
        return c

    def undo(self):
        """Undo last stroke."""
        if not self.undo_stack:
            return
        self.redo_stack.append(list(self.strokes))
        prev_strokes = self.undo_stack.pop()
        self.strokes = prev_strokes
        self._rebuild_canvas()

    def redo(self):
        """Redo last undone stroke."""
        if not self.redo_stack:
            return
        self.undo_stack.append(list(self.strokes))
        next_strokes = self.redo_stack.pop()
        self.strokes = next_strokes
        self._rebuild_canvas()

    def clear(self):
        """Clear all strokes."""
        if self.strokes:
            self.undo_stack.append(list(self.strokes))
            self.redo_stack.clear()
        self.strokes.clear()
        self.current_stroke = None
        self.particles.clear()
        self.canvas[:] = 0

    def _rebuild_canvas(self):
        """Redraw all strokes from scratch (for undo/redo)."""
        self.canvas[:] = 0
        self.particles.clear()
        for stroke in self.strokes:
            prev_pt = None
            for pt in stroke.points:
                if prev_pt is not None:
                    self._draw_segment(self.canvas, prev_pt, pt, stroke)
                else:
                    self._draw_dot(self.canvas, pt, stroke)
                prev_pt = pt

    def resize(self, width: int, height: int):
        """Resize canvas, scaling existing content."""
        old = self.canvas.copy()
        self.canvas = np.zeros((height, width, 4), dtype=np.uint8)
        if old.size > 0:
            resized = cv2.resize(old, (width, height))
            self.canvas[:resized.shape[0], :resized.shape[1]] = resized
        self.width = width
        self.height = height

    def set_brush(self, brush: BrushType):
        self.brush_type = brush

    def set_color(self, color: Tuple[int, int, int]):
        self.brush_color = color

    def set_size(self, size: int):
        self.brush_size = max(1, min(50, size))

    def set_glow(self, glow: int):
        self.glow_intensity = max(0, min(10, glow))

    def save_drawing(self, path: str) -> bool:
        """Save the current drawing as a PNG with alpha channel."""
        try:
            img = self.get_canvas_image()
            cv2.imwrite(path, img)
            return True
        except Exception:
            return False

    def replace_last_stroke_points(self, clean_points: List[Tuple[int, int]]):
        """Replace the points of the last stroke with clean shape points and redraw."""
        if not self.strokes or not clean_points:
            return
        self.strokes[-1].points = clean_points
        self._rebuild_canvas()

    def load_drawing(self, path: str) -> bool:
        """Load a drawing from a PNG file."""
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                b, g, r = cv2.split(img)
                a = np.where(
                    (b == 0) & (g == 0) & (r == 0), 0, 255
                ).astype(np.uint8)
                img = cv2.merge([b, g, r, a])
            resized = cv2.resize(img, (self.width, self.height))
            self.canvas = resized
            return True
        except Exception:
            return False

