from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(unsafe_hash=True)
class Detection:
    """A single YOLO-format detection with pixel-space properties.

    Normalized coordinates are in [0, 1]. Pixel properties are derived
    from the image dimensions passed at construction time.
    """

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    image_width: int
    image_height: int

    @property
    def x1(self) -> int:
        return max(0, round((self.x_center - self.width / 2) * self.image_width))

    @property
    def y1(self) -> int:
        return max(0, round((self.y_center - self.height / 2) * self.image_height))

    @property
    def x2(self) -> int:
        return min(
            self.image_width,
            round((self.x_center + self.width / 2) * self.image_width),
        )

    @property
    def y2(self) -> int:
        return min(
            self.image_height,
            round((self.y_center + self.height / 2) * self.image_height),
        )

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_yolo(cls, line: str, image: Image.Image) -> Detection:
        """Parse a single YOLO-format line into a Detection."""
        parts = line.strip().split()
        w, h = image.size
        return cls(
            class_id=int(parts[0]),
            x_center=round(float(parts[1]), 6),
            y_center=round(float(parts[2]), 6),
            width=round(float(parts[3]), 6),
            height=round(float(parts[4]), 6),
            image_width=w,
            image_height=h,
        )

    @classmethod
    def load_txt(cls, path: Path, image: Image.Image) -> list[Detection]:
        """Load all detections from a YOLO-format txt file."""
        if not path.exists():
            return []

        detections = []
        for line in path.read_text().strip().splitlines():
            detections.append(cls.from_yolo(line, image))
        return detections
