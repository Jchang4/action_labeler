from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from pydantic import BaseModel


class ActionResponse(BaseModel):
    """Base response model for action labeling.

    All prompt response models should inherit from this to ensure
    the ``action`` field is always present.
    """

    action: str


@dataclass
class LabelResult:
    """Standardized output from a labeler's label() method.

    Pairs the extracted action string with the full VLM response.
    """

    action: str
    response: ActionResponse | str


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
    segments: list[float] | None = field(default=None, hash=False, compare=False)

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

    @classmethod
    def from_segment_line(cls, line: str, image: Image.Image) -> Detection:
        """Parse a YOLO segment line into a Detection with segment data.

        Segment format: ``class_id x1 y1 x2 y2 ... xn yn`` (normalized).
        The bounding box is derived from the polygon extents.
        """
        parts = line.strip().split()
        w, h = image.size
        class_id = int(parts[0])
        coords = [round(float(v), 6) for v in parts[1:]]
        xs = coords[0::2]
        ys = coords[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return cls(
            class_id=class_id,
            x_center=round((min_x + max_x) / 2, 6),
            y_center=round((min_y + max_y) / 2, 6),
            width=round(max_x - min_x, 6),
            height=round(max_y - min_y, 6),
            image_width=w,
            image_height=h,
            segments=coords,
        )

    @classmethod
    def load_segments_txt(cls, path: Path, image: Image.Image) -> list[Detection]:
        """Load detections from a YOLO segment-format txt file."""
        if not path.exists():
            return []

        detections = []
        for line in path.read_text().strip().splitlines():
            detections.append(cls.from_segment_line(line, image))
        return detections
