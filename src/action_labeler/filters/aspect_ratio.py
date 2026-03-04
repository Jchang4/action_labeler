from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from action_labeler.filters.base import BaseFilter

if TYPE_CHECKING:
    from action_labeler.types import Detection


class AspectRatioFilter(BaseFilter):
    """Filter detections by bounding box aspect ratio (width / height).

    Useful for finding unusual shapes that may indicate difficult
    classifications:
    - Very tall objects (ratio < min_ratio)
    - Very wide objects (ratio > max_ratio)
    """

    def __init__(
        self,
        min_ratio: float = 0.0,
        max_ratio: float = float("inf"),
    ) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def filter(self, image: Image.Image, detections: list[Detection]) -> bool:
        """Keep the image only if all detections have aspect ratios in range."""
        return all(
            self.min_ratio <= self._aspect_ratio(d) <= self.max_ratio
            for d in detections
        )

    @staticmethod
    def _aspect_ratio(detection: Detection) -> float:
        if detection.height == 0:
            return float("inf")
        return detection.width / detection.height
