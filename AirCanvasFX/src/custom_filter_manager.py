"""
Custom Filter Manager - DoodleLens
Save, load, and manage user-created drawing-based face filters.
"""

import os
import json
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class CustomFilter:
    """Metadata for a user-created custom face filter."""
    name: str
    anchor: str
    scale: float
    rotation: float
    offset_x: int
    offset_y: int
    image_path: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CustomFilter":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


ANCHOR_OPTIONS = [
    "forehead",
    "above_head",
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
    "left_cheek",
    "right_cheek",
    "face_center",
    "chin",
]


class CustomFilterManager:
    """Manages saving, loading, and listing custom face filters."""

    def __init__(self, save_dir: str = "custom_filters"):
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self._index_path = os.path.join(self.save_dir, "index.json")
        self._filters: Dict[str, CustomFilter] = {}
        self._load_index()

    def _load_index(self):
        """Load the filter index from disk."""
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f)
                for name, meta in data.items():
                    self._filters[name] = CustomFilter.from_dict(meta)
            except Exception:
                self._filters = {}

    def _save_index(self):
        """Persist the filter index to disk."""
        with open(self._index_path, "w") as f:
            json.dump(
                {name: cf.to_dict() for name, cf in self._filters.items()},
                f, indent=2
            )

    def create_filter_from_canvas(self,
                                   name: str,
                                   canvas_bgra: np.ndarray,
                                   anchor: str = "forehead",
                                   scale: float = 1.0,
                                   rotation: float = 0.0,
                                   offset: Tuple[int, int] = (0, 0),
                                   description: str = "") -> Optional[CustomFilter]:
        """
        Create a custom filter from a BGRA drawing canvas.
        Crops the non-empty region and saves it as a transparent PNG.
        """
        # Find bounding box of non-transparent content
        if canvas_bgra.shape[2] < 4:
            # Add alpha if needed
            alpha = np.ones(canvas_bgra.shape[:2], dtype=np.uint8) * 255
            canvas_bgra = np.dstack([canvas_bgra, alpha])

        alpha_channel = canvas_bgra[:, :, 3]
        rows = np.any(alpha_channel > 10, axis=1)
        cols = np.any(alpha_channel > 10, axis=0)

        if not rows.any() or not cols.any():
            return None  # Empty canvas

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding
        pad = 20
        rmin = max(0, rmin - pad)
        rmax = min(canvas_bgra.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(canvas_bgra.shape[1] - 1, cmax + pad)

        cropped = canvas_bgra[rmin:rmax + 1, cmin:cmax + 1]

        # Save image
        safe_name = "".join(c for c in name if c.isalnum() or c in "_ -").strip()
        if not safe_name:
            safe_name = f"filter_{len(self._filters)}"

        img_path = os.path.join(self.save_dir, f"{safe_name}.png")
        cv2.imwrite(img_path, cropped)

        cf = CustomFilter(
            name=name,
            anchor=anchor,
            scale=scale,
            rotation=rotation,
            offset_x=offset[0],
            offset_y=offset[1],
            image_path=img_path,
            description=description,
        )

        self._filters[name] = cf
        self._save_index()
        return cf

    def load_overlay(self, name: str) -> Optional[np.ndarray]:
        """Load the BGRA image for a named custom filter."""
        cf = self._filters.get(name)
        if cf is None or not os.path.exists(cf.image_path):
            return None
        img = cv2.imread(cf.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            a = np.where(
                (b == 0) & (g == 0) & (r == 0), 0, 255
            ).astype(np.uint8)
            img = cv2.merge([b, g, r, a])
        return img

    def get_filter(self, name: str) -> Optional[CustomFilter]:
        return self._filters.get(name)

    def list_filters(self) -> List[CustomFilter]:
        return list(self._filters.values())

    def delete_filter(self, name: str) -> bool:
        cf = self._filters.pop(name, None)
        if cf is None:
            return False
        if os.path.exists(cf.image_path):
            os.remove(cf.image_path)
        self._save_index()
        return True

    def update_filter_metadata(self, name: str, **kwargs) -> bool:
        cf = self._filters.get(name)
        if cf is None:
            return False
        for key, val in kwargs.items():
            if hasattr(cf, key):
                setattr(cf, key, val)
        self._save_index()
        return True
