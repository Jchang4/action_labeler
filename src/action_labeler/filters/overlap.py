from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from PIL import Image

from action_labeler.filters.base import BaseFilter

if TYPE_CHECKING:
    from action_labeler.types import Detection


class OverlapFilter(BaseFilter):
    """Filter based on how crowded/overlapping the detection area is.

    Measures the maximum Intersection over Union (IoU) between any pair of
    detections. Crowded scenes with overlapping detections are harder to
    classify due to occlusion and visual clutter.
    """

    def __init__(self, max_iou: float = 0.5) -> None:
        self.max_iou = max_iou

    def filter(self, image: Image.Image, detections: list[Detection]) -> bool:
        """Keep the image only if no pair of detections exceeds max_iou."""
        for a, b in combinations(detections, 2):
            if self._iou(a, b) > self.max_iou:
                return False
        return True

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - intersection
        if union == 0:
            return 0.0

        return intersection / union
