import supervision as sv
from PIL import Image

from action_labeler.v2.base import IFilter


class SingleDetectionFilter(IFilter):
    """Filter images with more than one detection."""

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
    ) -> bool:
        return len(detections.xyxy) == 1


class MaxDetectionsFilter(IFilter):
    """Filter images with more than a certain number of detections."""

    max_detections: int

    def __init__(self, max_detections: int = 99999999):
        self.max_detections = max_detections

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
    ) -> bool:
        return len(detections.xyxy) <= self.max_detections
