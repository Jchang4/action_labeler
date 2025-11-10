from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter


class AspectRatioFilter(IFilter):
    """Filter detections by bounding box aspect ratio.

    Aspect ratio is calculated as width / height. This filter is useful for
    finding unusual shapes that may indicate difficult classifications:
    - Very tall objects (ratio < min_ratio)
    - Very wide objects (ratio > max_ratio)

    Args:
        min_ratio: Minimum allowed aspect ratio (width/height)
        max_ratio: Maximum allowed aspect ratio (width/height)

    Examples:
        # Only roughly square objects (0.8 <= w/h <= 1.2)
        AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)

        # Exclude very tall objects (keep w/h >= 0.3)
        AspectRatioFilter(min_ratio=0.3, max_ratio=100.0)

        # Exclude very wide objects (keep w/h <= 3.0)
        AspectRatioFilter(min_ratio=0.0, max_ratio=3.0)
    """

    min_ratio: float
    max_ratio: float

    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 10.0):
        if min_ratio < 0:
            raise ValueError("min_ratio must be non-negative")
        if max_ratio < 0:
            raise ValueError("max_ratio must be non-negative")
        if min_ratio > max_ratio:
            raise ValueError("min_ratio must be <= max_ratio")

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        xyxy = detections.xyxy[index]
        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1

        # Avoid division by zero
        if height == 0:
            return False

        aspect_ratio = float(width) / float(height)
        return self.min_ratio <= aspect_ratio <= self.max_ratio
