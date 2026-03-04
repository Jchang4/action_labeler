from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from action_labeler.filters.base import BaseFilter

if TYPE_CHECKING:
    from action_labeler.types import Detection


class DetectionCountFilter(BaseFilter):
    """Keep images with min_count <= number of detections <= max_count."""

    def __init__(
        self,
        min_count: int = 0,
        max_count: int | None = None,
    ) -> None:
        self.min_count = min_count
        self.max_count = max_count

    def filter(self, image: Image.Image, detections: list[Detection]) -> bool:
        n = len(detections)
        if n < self.min_count:
            return False
        if self.max_count is not None and n > self.max_count:
            return False
        return True
