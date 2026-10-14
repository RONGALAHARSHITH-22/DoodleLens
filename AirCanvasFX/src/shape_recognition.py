"""
Shape Recognition Module - DoodleLens
Recognizes freehand shapes from drawn strokes.
"""

import cv2
import numpy as np
import math
from typing import Optional, List, Tuple
from enum import Enum, auto


class ShapeType(Enum):
    UNKNOWN = "unknown"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    STAR = "star"
    HEART = "heart"
    ARROW = "arrow"
    INFINITY = "infinity"
    LINE = "line"


class ShapeRecognizer:
    """
    Recognizes freehand shapes from a list of stroke points.
    Uses contour approximation and geometric analysis.
    """

    def __init__(self, min_points: int = 10, shape_threshold: float = 0.8):
        self.min_points = min_points
        self.shape_threshold = shape_threshold

    def recognize(self, points: List[Tuple[int, int]]) -> ShapeType:
        """Attempt to recognize the shape from the list of points."""
        if len(points) < self.min_points:
            return ShapeType.UNKNOWN

        pts_array = np.array(points, dtype=np.int32)

        # ── Early line check: detect before area-based metrics ─────────────
        if len(pts_array) > 5:
            start_end_d = math.hypot(
                points[0][0] - points[-1][0],
                points[0][1] - points[-1][1]
            )
            _span = math.hypot(
                max(p[0] for p in points) - min(p[0] for p in points),
                max(p[1] for p in points) - min(p[1] for p in points)
            )
            if start_end_d >= _span * 0.6 or True:  # always try line fit
                try:
                    vx2, vy2, x02, y02 = cv2.fitLine(
                        pts_array, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx2, vy2 = float(vx2.flat[0]), float(vy2.flat[0])
                    x02, y02 = float(x02.flat[0]), float(y02.flat[0])
                    _total = sum(
                        abs((p[0] - x02) * vy2 - (p[1] - y02) * vx2)
                        for p in points
                    )
                    if _total / len(points) < 8.0:
                        return ShapeType.LINE
                except Exception:
                    pass

        # Compute contour metrics
        hull = cv2.convexHull(pts_array)
        hull_area = cv2.contourArea(hull)
        pts_area = cv2.contourArea(pts_array)

        if hull_area < 100:
            return ShapeType.UNKNOWN

        # Perimeter
        perimeter = cv2.arcLength(pts_array, True)

        # Circularity: 4π·area/perimeter²
        if perimeter > 0:
            circularity = 4 * math.pi * hull_area / (perimeter ** 2)
        else:
            return ShapeType.UNKNOWN

        # Solidity: area / hull_area
        solidity = pts_area / hull_area if hull_area > 0 else 0

        # Approximate polygon
        epsilon = 0.05 * perimeter
        approx = cv2.approxPolyDP(pts_array, epsilon, True)
        num_vertices = len(approx)

        # Check start/end proximity (closed shape)
        start_end_dist = math.hypot(
            points[0][0] - points[-1][0],
            points[0][1] - points[-1][1]
        )
        span = math.hypot(
            max(p[0] for p in points) - min(p[0] for p in points),
            max(p[1] for p in points) - min(p[1] for p in points)
        )
        is_closed = start_end_dist < span * 0.3

        # Circle: high circularity, closed
        if is_closed and circularity > 0.75:
            return ShapeType.CIRCLE

        # Triangle: 3 vertices, closed
        if is_closed and 3 <= num_vertices <= 4 and solidity > 0.75:
            return ShapeType.TRIANGLE

        # Rectangle: 4 vertices, closed, high solidity
        if is_closed and 4 <= num_vertices <= 6 and solidity > 0.80:
            # Check aspect ratio of bounding box
            rect = cv2.minAreaRect(pts_array)
            w, h = rect[1]
            if w > 0 and h > 0:
                ratio = max(w, h) / min(w, h)
                if ratio < 3.0:
                    return ShapeType.RECTANGLE

        # Star: many vertices, low solidity
        if is_closed and num_vertices >= 8 and solidity < 0.60:
            return ShapeType.STAR

        # Line: mostly straight (allow up to 15px average deviation)
        if not is_closed:
            if len(pts_array) > 5:
                vx, vy, x0, y0 = cv2.fitLine(pts_array, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy = float(vx), float(vy)
                x0, y0 = float(x0), float(y0)
                total_dist = 0.0
                for p in points:
                    dx = p[0] - x0
                    dy = p[1] - y0
                    cross = abs(dx * vy - dy * vx)
                    total_dist += cross
                avg_dist = total_dist / len(points)
                if avg_dist < 15:
                    return ShapeType.LINE

        return ShapeType.UNKNOWN

    def get_clean_shape(self, shape: ShapeType,
                         points: List[Tuple[int, int]],
                         canvas_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Generate a clean geometric shape overlay (BGRA) at the same position
        as the freehand drawing.
        """
        if shape == ShapeType.UNKNOWN or not points:
            return None

        w, h = canvas_size
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        pts_array = np.array(points, dtype=np.int32)
        color = (0, 255, 255, 255)  # Cyan

        if shape == ShapeType.CIRCLE:
            (cx, cy), radius = cv2.minEnclosingCircle(pts_array)
            cv2.circle(overlay, (int(cx), int(cy)), int(radius), color, 2)

        elif shape == ShapeType.TRIANGLE:
            hull = cv2.convexHull(pts_array)
            approx = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
            if len(approx) >= 3:
                cv2.polylines(overlay, [approx[:3]], True, color, 2)

        elif shape == ShapeType.RECTANGLE:
            rect = cv2.minAreaRect(pts_array)
            box = cv2.boxPoints(rect).astype(int)
            cv2.polylines(overlay, [box], True, color, 2)

        elif shape == ShapeType.LINE:
            if len(points) >= 2:
                cv2.line(overlay, points[0], points[-1], color, 2)

        else:
            # For complex shapes: draw convex hull
            hull = cv2.convexHull(pts_array)
            cv2.polylines(overlay, [hull], True, color, 2)

        return overlay

    def get_clean_points(self, shape: ShapeType,
                         points: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """
        Generate a list of clean geometric vertex points for snapping
        the freehand stroke into a clean shape.
        """
        if shape == ShapeType.UNKNOWN or not points or len(points) < 2:
            return None

        pts_array = np.array(points, dtype=np.int32)

        if shape == ShapeType.CIRCLE:
            (cx, cy), radius = cv2.minEnclosingCircle(pts_array)
            r = max(5.0, float(radius))
            num_pts = 36
            return [
                (int(cx + r * math.cos(2 * math.pi * i / num_pts)),
                 int(cy + r * math.sin(2 * math.pi * i / num_pts)))
                for i in range(num_pts + 1)
            ]

        elif shape == ShapeType.TRIANGLE:
            hull = cv2.convexHull(pts_array)
            approx = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
            if len(approx) >= 3:
                pts = [tuple(map(int, p[0])) for p in approx[:3]]
                pts.append(pts[0])  # Close triangle
                return pts

        elif shape == ShapeType.RECTANGLE:
            rect = cv2.minAreaRect(pts_array)
            box = cv2.boxPoints(rect).astype(int)
            pts = [tuple(map(int, p)) for p in box]
            pts.append(pts[0])  # Close box
            return pts

        elif shape == ShapeType.LINE:
            return [tuple(map(int, points[0])), tuple(map(int, points[-1]))]

        return None

