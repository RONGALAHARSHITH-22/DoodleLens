"""
Face Filter Engine - DoodleLens
Renders built-in AR face filters anchored to facial landmarks.
"""

import cv2
import numpy as np
import math
import random
import time
from typing import Dict, Optional, Tuple, List
from enum import Enum, auto


class FilterType(Enum):
    NONE = "none"
    NEON_AURA = "neon_aura"
    CYBERPUNK_MASK = "cyberpunk_mask"
    ANGEL_MODE = "angel_mode"
    DEMON_MODE = "demon_mode"
    STAR_EYES = "star_eyes"
    HEART_EYES = "heart_eyes"
    GALAXY_FACE = "galaxy_face"
    RAINBOW_CROWN = "rainbow_crown"
    SKETCH_FACE = "sketch_face"
    CUSTOM = "custom"


FILTER_NAMES = {
    FilterType.NONE: "No Filter",
    FilterType.NEON_AURA: "Neon Aura",
    FilterType.CYBERPUNK_MASK: "Cyberpunk Mask",
    FilterType.ANGEL_MODE: "Angel Mode",
    FilterType.DEMON_MODE: "Demon Mode",
    FilterType.STAR_EYES: "Star Eyes",
    FilterType.HEART_EYES: "Heart Eyes",
    FilterType.GALAXY_FACE: "Galaxy Face",
    FilterType.RAINBOW_CROWN: "Rainbow Crown",
    FilterType.SKETCH_FACE: "Sketch Face",
    FilterType.CUSTOM: "Custom Drawing",
}


class FaceParticle:
    """Particle anchored near face."""
    __slots__ = ["ox", "oy", "vx", "vy", "life", "max_life", "color", "size"]

    def __init__(self, ox: float, oy: float, color: Tuple, max_life: int = 40):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 2.5)
        self.ox = ox
        self.oy = oy
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 1.5  # float upward
        self.life = max_life
        self.max_life = max_life
        self.color = color
        self.size = random.randint(2, 5)

    def update(self) -> bool:
        self.ox += self.vx
        self.oy += self.vy
        self.life -= 1
        return self.life > 0


class FaceFilterEngine:
    """Renders AR face filters anchored to MediaPipe face landmarks."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.active_filter = FilterType.NONE
        self.filter_intensity = 1.0
        self.particles: List[FaceParticle] = []
        self._hue = 0.0
        self._anim_t = 0.0
        self._frame = 0
        self._custom_overlay: Optional[np.ndarray] = None
        self._custom_anchor = "forehead"
        self._custom_scale = 1.0
        self._custom_rotation = 0.0
        self._custom_offset = (0, 0)

    def set_filter(self, filter_type: FilterType):
        self.active_filter = filter_type
        self.particles.clear()

    def set_custom_overlay(self, overlay: np.ndarray, anchor: str = "forehead",
                            scale: float = 1.0, rotation: float = 0.0,
                            offset: Tuple[int, int] = (0, 0)):
        self.active_filter = FilterType.CUSTOM
        self._custom_overlay = overlay
        self._custom_anchor = anchor
        self._custom_scale = scale
        self._custom_rotation = rotation
        self._custom_offset = offset

    def render(self, frame: np.ndarray,
               face_data: Optional[Dict]) -> np.ndarray:
        """Apply the active filter onto frame using face_data."""
        self._hue = (self._hue + 3) % 360
        self._anim_t += 0.1
        self._frame += 1
        self.particles = [p for p in self.particles if p.update()]

        if self.active_filter == FilterType.NONE or face_data is None:
            return frame

        named = face_data.get("named_landmarks", {})
        face_cx, face_cy = face_data.get("face_center", (0, 0))
        face_w = face_data.get("face_width", 100)
        face_h = face_data.get("face_height", 100)
        head_tilt = face_data.get("head_tilt", 0.0)
        bbox = face_data.get("face_bbox", (0, 0, 100, 100))

        out = frame.copy()

        try:
            if self.active_filter == FilterType.NEON_AURA:
                out = self._render_neon_aura(out, face_data)
            elif self.active_filter == FilterType.CYBERPUNK_MASK:
                out = self._render_cyberpunk(out, face_data)
            elif self.active_filter == FilterType.ANGEL_MODE:
                out = self._render_angel(out, face_data)
            elif self.active_filter == FilterType.DEMON_MODE:
                out = self._render_demon(out, face_data)
            elif self.active_filter == FilterType.STAR_EYES:
                out = self._render_star_eyes(out, face_data)
            elif self.active_filter == FilterType.HEART_EYES:
                out = self._render_heart_eyes(out, face_data)
            elif self.active_filter == FilterType.GALAXY_FACE:
                out = self._render_galaxy(out, face_data)
            elif self.active_filter == FilterType.RAINBOW_CROWN:
                out = self._render_rainbow_crown(out, face_data)
            elif self.active_filter == FilterType.SKETCH_FACE:
                out = self._render_sketch(out, face_data)
            elif self.active_filter == FilterType.CUSTOM:
                out = self._render_custom(out, face_data)
        except Exception:
            pass  # Fail gracefully

        return out

    # ── Helper: draw glow circle ──────────────────────────────────────────
    def _glow_circle(self, frame, center, radius, color, layers=3):
        for i in range(layers, 0, -1):
            alpha = int(80 / i)
            overlay = frame.copy()
            cv2.circle(overlay, center, radius + i * 5, color, 2)
            cv2.addWeighted(overlay, alpha / 255.0, frame,
                            1 - alpha / 255.0, 0, frame)
        cv2.circle(frame, center, radius, color, 2)
        return frame

    def _glow_line(self, frame, p1, p2, color, thickness=2, layers=3):
        for i in range(layers, 0, -1):
            overlay = frame.copy()
            cv2.line(overlay, p1, p2, color, thickness + i * 2)
            alpha = 60 // i
            cv2.addWeighted(overlay, alpha / 255.0, frame,
                            1 - alpha / 255.0, 0, frame)
        cv2.line(frame, p1, p2, color, thickness)
        return frame

    @staticmethod
    def _hsv_to_bgr(h, s=1.0, v=1.0):
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60: r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        return (int((b + m)*255), int((g + m)*255), int((r + m)*255))

    def _draw_star(self, frame, cx, cy, r, color, num_points=5):
        """Draw a star polygon."""
        pts = []
        for i in range(num_points * 2):
            angle = math.pi / num_points * i - math.pi / 2
            rad = r if i % 2 == 0 else r // 2
            pts.append((
                int(cx + rad * math.cos(angle)),
                int(cy + rad * math.sin(angle))
            ))
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(frame, [pts], color)
        return frame

    def _draw_heart(self, frame, cx, cy, size, color):
        """Draw a heart shape."""
        overlay = frame.copy()
        pts = []
        for t in np.linspace(0, 2 * math.pi, 60):
            x = int(cx + size * 16 * math.sin(t)**3 / 16)
            y = int(cy - size * (13 * math.cos(t) - 5 * math.cos(2*t) -
                                  2 * math.cos(3*t) - math.cos(4*t)) / 16)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        return frame

    def _overlay_alpha(self, frame, overlay_bgra, pos, scale=1.0, angle=0.0):
        """
        Composite a BGRA overlay image onto frame at pos (center).
        """
        if overlay_bgra is None:
            return frame
        h_ov, w_ov = overlay_bgra.shape[:2]
        # Scale
        new_w = max(1, int(w_ov * scale))
        new_h = max(1, int(h_ov * scale))
        resized = cv2.resize(overlay_bgra, (new_w, new_h))

        # Rotate if needed
        if abs(angle) > 0.5:
            M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
            resized = cv2.warpAffine(resized, M, (new_w, new_h))

        # Place at center pos
        cx, cy = pos
        x1 = cx - new_w // 2
        y1 = cy - new_h // 2
        x2 = x1 + new_w
        y2 = y1 + new_h

        # Clip to frame
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        ox1 = fx1 - x1
        oy1 = fy1 - y1
        ox2 = ox1 + (fx2 - fx1)
        oy2 = oy1 + (fy2 - fy1)

        if fx2 <= fx1 or fy2 <= fy1:
            return frame

        alpha = resized[oy1:oy2, ox1:ox2, 3:4].astype(float) / 255.0
        for c in range(3):
            frame[fy1:fy2, fx1:fx2, c] = (
                frame[fy1:fy2, fx1:fx2, c].astype(float) * (1 - alpha[:, :, 0]) +
                resized[oy1:oy2, ox1:ox2, c].astype(float) * alpha[:, :, 0]
            ).astype(np.uint8)
        return frame

    # ── Filters ──────────────────────────────────────────────────────────

    def _render_neon_aura(self, frame, face_data):
        named = face_data["named_landmarks"]
        face_cx, face_cy = face_data["face_center"]
        face_w = face_data["face_width"]
        face_h = face_data["face_height"]

        # Pulsing aura ring
        pulse = int(10 * math.sin(self._anim_t * 2))
        r = int((face_w // 2 + 30 + pulse) * self.filter_intensity)
        color1 = self._hsv_to_bgr(self._hue, 1.0, 1.0)
        color2 = self._hsv_to_bgr((self._hue + 120) % 360, 1.0, 1.0)
        color3 = self._hsv_to_bgr((self._hue + 240) % 360, 1.0, 1.0)

        self._glow_circle(frame, (face_cx, face_cy), r, color1, 4)
        self._glow_circle(frame, (face_cx, face_cy), r + 20, color2, 3)
        self._glow_circle(frame, (face_cx, face_cy), r + 40, color3, 2)

        # Particles around face outline
        if self._frame % 3 == 0:
            for _ in range(3):
                angle = random.uniform(0, 2 * math.pi)
                px = face_cx + int((face_w // 2 + 20) * math.cos(angle))
                py = face_cy + int((face_h // 2 + 20) * math.sin(angle))
                c = self._hsv_to_bgr(random.uniform(0, 360))
                self.particles.append(FaceParticle(px, py, c))

        # Draw particles
        for p in self.particles:
            if 0 <= int(p.ox) < frame.shape[1] and 0 <= int(p.oy) < frame.shape[0]:
                ratio = p.life / p.max_life
                cv2.circle(frame, (int(p.ox), int(p.oy)),
                           max(1, int(p.size * ratio)), p.color, -1)

        return frame

    def _render_cyberpunk(self, frame, face_data):
        named = face_data["named_landmarks"]
        face_cx, face_cy = face_data["face_center"]
        face_w = face_data["face_width"]
        face_h = face_data["face_height"]
        head_tilt = face_data.get("head_tilt", 0.0)

        cyan = (255, 255, 0)
        magenta = (255, 0, 200)
        green = (0, 255, 100)

        # Visor lines across eyes
        if "left_eye_center" in named and "right_eye_center" in named:
            lec = named["left_eye_center"]
            rec = named["right_eye_center"]
            eye_y = (lec[1] + rec[1]) // 2
            x_left = face_data["face_bbox"][0] - 10
            x_right = face_data["face_bbox"][0] + face_w + 10
            self._glow_line(frame, (x_left, eye_y), (x_right, eye_y), cyan, 2, 3)
            self._glow_line(frame, (x_left, eye_y - 8), (x_right, eye_y - 8), cyan, 1, 2)
            self._glow_line(frame, (x_left, eye_y + 8), (x_right, eye_y + 8), cyan, 1, 2)

        # Triangle on forehead
        if "forehead" in named and "left_eyebrow_outer" in named and \
                "right_eyebrow_outer" in named:
            fore = named["forehead"]
            lb = named["left_eyebrow_outer"]
            rb = named["right_eyebrow_outer"]
            pts = np.array([fore, lb, rb], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (*magenta, 80))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [pts], True, magenta, 1)

        # Nose bridge line
        if "nose_bridge" in named and "nose_tip" in named:
            self._glow_line(frame, named["nose_bridge"], named["nose_tip"],
                             green, 1, 2)

        # Corner brackets on face
        bx, by, bw, bh = face_data["face_bbox"]
        sz = 15
        for cx2, cy2 in [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]:
            dx = 1 if cx2 == bx else -1
            dy = 1 if cy2 == by else -1
            cv2.line(frame, (cx2, cy2), (cx2 + dx * sz, cy2), cyan, 2)
            cv2.line(frame, (cx2, cy2), (cx2, cy2 + dy * sz), cyan, 2)

        return frame

    def _render_angel(self, frame, face_data):
        named = face_data["named_landmarks"]
        face_cx, face_cy = face_data["face_center"]
        face_w = face_data["face_width"]
        head_tilt = face_data.get("head_tilt", 0.0)

        if "forehead" not in named:
            return frame

        fore = named["forehead"]
        halo_cx = fore[0]
        halo_cy = fore[1] - int(face_w * 0.35)
        halo_r = int(face_w * 0.4 * self.filter_intensity)

        # Halo with golden glow
        gold = (0, 200, 255)
        white = (255, 255, 255)
        pulse = int(5 * math.sin(self._anim_t * 3))

        for layer in range(6, 0, -1):
            a = int(50 / layer)
            overlay = frame.copy()
            cv2.ellipse(overlay, (halo_cx, halo_cy),
                        (halo_r + pulse + layer * 3, (halo_r + pulse) // 3 + layer),
                        head_tilt, 0, 360, gold, 3)
            cv2.addWeighted(overlay, a / 255.0, frame, 1 - a / 255.0, 0, frame)

        cv2.ellipse(frame, (halo_cx, halo_cy),
                    (halo_r + pulse, (halo_r + pulse) // 3),
                    head_tilt, 0, 360, gold, 3)

        # Floating sparkles
        if self._frame % 5 == 0:
            for _ in range(2):
                angle = random.uniform(0, 2 * math.pi)
                px = halo_cx + int(halo_r * math.cos(angle))
                py = halo_cy + int((halo_r // 3) * math.sin(angle))
                self.particles.append(FaceParticle(px, py, white, 60))

        for p in self.particles:
            if 0 <= int(p.ox) < frame.shape[1] and 0 <= int(p.oy) < frame.shape[0]:
                ratio = p.life / p.max_life
                # Draw star sparkle
                r2 = max(1, int(4 * ratio))
                cv2.circle(frame, (int(p.ox), int(p.oy)), r2, white, -1)

        return frame

    def _render_demon(self, frame, face_data):
        named = face_data["named_landmarks"]
        face_w = face_data["face_width"]
        face_h = face_data["face_height"]
        head_tilt = face_data.get("head_tilt", 0.0)

        if "left_eyebrow_outer" not in named or "right_eyebrow_outer" not in named:
            return frame

        lb = named["left_eyebrow_outer"]
        rb = named["right_eyebrow_outer"]
        horn_h = int(face_h * 0.45 * self.filter_intensity)
        horn_w = int(face_w * 0.12)
        red = (0, 0, 220)
        dark_red = (0, 0, 160)

        # Left horn
        for dx, c in [(0, red), (-horn_w // 3, dark_red)]:
            lh_pts = np.array([
                [lb[0] + dx, lb[1]],
                [lb[0] + dx - horn_w // 2, lb[1] - horn_h],
                [lb[0] + dx + horn_w // 2, lb[1]],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [lh_pts], c)
            cv2.polylines(frame, [lh_pts], True, (0, 0, 255), 2)

        # Right horn
        for dx, c in [(0, red), (horn_w // 3, dark_red)]:
            rh_pts = np.array([
                [rb[0] + dx, rb[1]],
                [rb[0] + dx, rb[1] - horn_h],
                [rb[0] + dx + horn_w, rb[1]],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [rh_pts], c)
            cv2.polylines(frame, [rh_pts], True, (0, 0, 255), 2)

        # Glowing eye effect
        for eye_key in ["left_eye_center", "right_eye_center"]:
            if eye_key in named:
                ec = named[eye_key]
                overlay = frame.copy()
                cv2.circle(overlay, ec, int(face_w * 0.06), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Ember particles
        if self._frame % 4 == 0:
            for horn_pt in [lb, rb]:
                c = random.choice([(0, 0, 255), (0, 100, 255), (0, 50, 200)])
                self.particles.append(FaceParticle(horn_pt[0], horn_pt[1] - horn_h, c, 30))

        for p in self.particles:
            if 0 <= int(p.ox) < frame.shape[1] and 0 <= int(p.oy) < frame.shape[0]:
                ratio = p.life / p.max_life
                cv2.circle(frame, (int(p.ox), int(p.oy)),
                           max(1, int(3 * ratio)), p.color, -1)

        return frame

    def _render_star_eyes(self, frame, face_data):
        named = face_data["named_landmarks"]
        face_w = face_data["face_width"]
        eye_size = int(face_w * 0.12 * self.filter_intensity)

        anim_angle = self._anim_t * 30  # rotating stars

        for key, color in [("left_eye_center", (0, 220, 255)),
                            ("right_eye_center", (255, 100, 220))]:
            if key not in named:
                key = "left_eye_inner" if "left" in key else "right_eye_inner"
                if key not in named:
                    continue
            ec = named[key]
            # Rotating star
            M = cv2.getRotationMatrix2D(ec, anim_angle, 1.0)
            tmp = np.zeros_like(frame)
            self._draw_star(tmp, ec[0], ec[1], eye_size, color)
            # Glow
            blurred = cv2.GaussianBlur(tmp, (15, 15), 0)
            frame = cv2.add(frame, blurred)
            frame = cv2.add(frame, tmp)

        return frame

    def _render_heart_eyes(self, frame, face_data):
        named = face_data["named_landmarks"]
        face_w = face_data["face_width"]
        h_size = int(face_w * 0.06 * self.filter_intensity)

        pulse = 1.0 + 0.2 * math.sin(self._anim_t * 4)

        for key in ["left_eye_center", "right_eye_center"]:
            if key not in named:
                continue
            ec = named[key]
            size = int(h_size * pulse)
            # Draw animated bouncing heart
            tmp = np.zeros_like(frame)
            self._draw_heart(tmp, ec[0], ec[1], size, (60, 20, 220))
            blurred = cv2.GaussianBlur(tmp, (11, 11), 0)
            frame = cv2.add(frame, blurred)
            self._draw_heart(frame, ec[0], ec[1], size, (80, 30, 240))

        return frame

    def _render_galaxy(self, frame, face_data):
        named = face_data["named_landmarks"]
        face_cx, face_cy = face_data["face_center"]
        face_w = face_data["face_width"]
        face_h = face_data["face_height"]

        # Dark overlay inside face
        if face_data.get("face_bbox"):
            bx, by, bw, bh = face_data["face_bbox"]
            overlay = frame.copy()
            cv2.ellipse(overlay, (face_cx, face_cy),
                        (bw // 2, bh // 2), 0, 0, 360, (20, 10, 30), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Galaxy particles / cosmic trails
        if self._frame % 2 == 0:
            for _ in range(5):
                angle = random.uniform(0, 2 * math.pi)
                r_offset = random.uniform(0.3, 0.7)
                px = face_cx + int(face_w * r_offset * math.cos(angle))
                py = face_cy + int(face_h * r_offset * math.sin(angle))
                c = self._hsv_to_bgr(random.uniform(200, 320))
                self.particles.append(FaceParticle(px, py, c, 50))

        for p in self.particles:
            if 0 <= int(p.ox) < frame.shape[1] and 0 <= int(p.oy) < frame.shape[0]:
                ratio = p.life / p.max_life
                cv2.circle(frame, (int(p.ox), int(p.oy)),
                           max(1, int(p.size * ratio)), p.color, -1)

        # Rings
        for i in range(3):
            r = int((face_w * 0.4 + i * 25) * self.filter_intensity)
            angle_offset = self._anim_t * 20 + i * 30
            color = self._hsv_to_bgr((self._hue + i * 60) % 360)
            cv2.ellipse(frame, (face_cx, face_cy),
                        (r, r // 3), angle_offset, 0, 360, color, 1)

        return frame

    def _render_rainbow_crown(self, frame, face_data):
        named = face_data["named_landmarks"]
        face_w = face_data["face_width"]
        head_tilt = face_data.get("head_tilt", 0.0)

        if "forehead" not in named:
            return frame

        fore = named["forehead"]
        crown_y = fore[1] - int(face_w * 0.15)
        crown_cx = fore[0]
        crown_w = int(face_w * 0.8 * self.filter_intensity)
        crown_h = int(face_w * 0.25)

        # Crown prongs
        num_prongs = 5
        prong_colors = [self._hsv_to_bgr(i * 60) for i in range(num_prongs)]
        crown_pts = []
        for i in range(num_prongs):
            t = i / (num_prongs - 1) - 0.5
            bx = int(crown_cx + t * crown_w)
            tip_h = crown_h if i % 2 == 0 else crown_h // 2
            crown_pts.append((bx, crown_y - tip_h))
            crown_pts.append((bx, crown_y))

        # Draw crown shape
        all_pts = []
        for i in range(num_prongs):
            t = i / (num_prongs - 1) - 0.5
            bx = int(crown_cx + t * crown_w)
            tip_h = crown_h if i % 2 == 0 else crown_h // 2
            all_pts.append([bx, crown_y - tip_h])
        all_pts.append([crown_cx + crown_w // 2, crown_y])
        all_pts.append([crown_cx - crown_w // 2, crown_y])
        pts_array = np.array(all_pts, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_array], self._hsv_to_bgr(self._hue))
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.polylines(frame, [pts_array], True, (255, 255, 255), 2)

        # Rainbow outline
        for i, color in enumerate(prong_colors):
            t = i / (num_prongs - 1) - 0.5
            bx = int(crown_cx + t * crown_w)
            cv2.circle(frame, (bx, crown_y - crown_h if i % 2 == 0 else crown_y - crown_h // 2),
                       5, color, -1)

        return frame

    def _render_sketch(self, frame, face_data):
        """Stylized contour/sketch overlay following face landmarks."""
        named = face_data["named_landmarks"]
        face_w = face_data["face_width"]

        # Groups of connected points
        eye_pts_left = []
        eye_pts_right = []
        for key in ["left_eye_outer", "left_eye_inner"]:
            if key in named:
                eye_pts_left.append(named[key])
        for key in ["right_eye_inner", "right_eye_outer"]:
            if key in named:
                eye_pts_right.append(named[key])

        pencil_color = (200, 220, 255)  # light blue-ish
        line_thickness = 2

        # Draw eye outlines
        for pts in [eye_pts_left, eye_pts_right]:
            if len(pts) >= 2:
                for j in range(len(pts) - 1):
                    self._glow_line(frame, pts[j], pts[j + 1], pencil_color, line_thickness, 2)

        # Eyebrows
        for key in ["left_eyebrow_outer", "right_eyebrow_outer"]:
            if key in named:
                pt = named[key]
                npt = (pt[0], pt[1] - int(face_w * 0.04))
                self._glow_line(frame, pt, npt, pencil_color, line_thickness, 2)

        # Nose
        if "nose_bridge" in named and "nose_tip" in named:
            self._glow_line(frame, named["nose_bridge"], named["nose_tip"],
                             pencil_color, line_thickness, 2)

        # Mouth
        if "mouth_left" in named and "mouth_right" in named:
            self._glow_line(frame, named["mouth_left"], named["mouth_right"],
                             pencil_color, line_thickness, 2)

        # Face outline (ellipse)
        if face_data.get("face_bbox"):
            bx, by, bw, bh = face_data["face_bbox"]
            cx, cy = face_data["face_center"]
            cv2.ellipse(frame, (cx, cy), (bw // 2, bh // 2),
                        face_data.get("head_tilt", 0), 0, 360, pencil_color, 1)

        return frame

    def _render_custom(self, frame, face_data):
        """Render a custom user-drawn overlay anchored to a face landmark."""
        if self._custom_overlay is None:
            return frame

        named = face_data["named_landmarks"]
        face_w = face_data["face_width"]
        face_h = face_data["face_height"]
        face_cx, face_cy = face_data["face_center"]
        head_tilt = face_data.get("head_tilt", 0.0)

        # Determine anchor position
        anchor_map = {
            "forehead": named.get("forehead", (face_cx, face_cy - face_h // 3)),
            "above_head": (face_cx, face_data["face_bbox"][1] - int(face_h * 0.2))
                if face_data.get("face_bbox") else (face_cx, face_cy - face_h),
            "left_eye": named.get("left_eye_center",
                                  named.get("left_eye_outer", (face_cx, face_cy))),
            "right_eye": named.get("right_eye_center",
                                   named.get("right_eye_outer", (face_cx, face_cy))),
            "nose": named.get("nose_tip", (face_cx, face_cy)),
            "mouth": named.get("mouth_top", (face_cx, face_cy + face_h // 4)),
            "left_cheek": named.get("left_cheek", (face_cx - face_w // 3, face_cy)),
            "right_cheek": named.get("right_cheek", (face_cx + face_w // 3, face_cy)),
            "face_center": (face_cx, face_cy),
            "chin": named.get("chin", (face_cx, face_cy + face_h // 2)),
        }

        pos = anchor_map.get(self._custom_anchor, (face_cx, face_cy))
        # Apply user offset
        pos = (pos[0] + self._custom_offset[0], pos[1] + self._custom_offset[1])

        # Auto-scale by face size
        scale = self._custom_scale * (face_w / 200.0)

        frame = self._overlay_alpha(frame, self._custom_overlay, pos,
                                     scale=scale,
                                     angle=head_tilt + self._custom_rotation)
        return frame
