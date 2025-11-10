from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter


class SingleDetectionFilter(IFilter):
    """Filter images with more than one detection."""

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        return len(detections.xyxy) == 1


class MaxDetectionsFilter(IFilter):
    """Filter images with more than a certain number of detections."""

    max_detections: int

    def __init__(self, max_detections: int):
        if max_detections <= 0:
            raise ValueError("max_detections must be greater than 0")
        self.max_detections = max_detections

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        return len(detections.xyxy) <= self.max_detections


class MinDetectionsFilter(IFilter):
    """Filter images with less than a certain number of detections."""

    min_detections: int

    def __init__(self, min_detections: int):
        if min_detections <= 0:
            raise ValueError("min_detections must be greater than 0")
        self.min_detections = min_detections

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        return len(detections.xyxy) >= self.min_detections
