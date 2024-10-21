from PIL import Image
import supervision as sv

from .base import BaseImageFilter


class SingleDetectionFilter(BaseImageFilter):
    """Filter out images that have more than one detection."""

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
    ) -> bool:
        return len(detections.xyxy) == 1


class MultipleDetectionsFilter(BaseImageFilter):
    """Filter out images that have more than a certain number of detections."""

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


class MinImageSizeFilter(BaseImageFilter):
    """Filter out images that are too small."""

    min_size: int

    def __init__(self, min_size: int):
        self.min_size = min_size

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
    ) -> bool:
        return image.width >= self.min_size and image.height >= self.min_size


class SmallDetectionsFilter(BaseImageFilter):
    """Filter out detections that are too small."""

    min_area: float
    size_approve: int

    def __init__(self, min_area: float = 0.1, size_approve: int = 300):
        self.min_area = min_area
        self.size_approve = size_approve

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
    ) -> bool:
        xyxy = detections.xyxy[index]
        x1, y1, x2, y2 = xyxy
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        image_area = image.width * image.height

        # For detections in very large images, we approve
        # detections that are larger than size_approve
        if box_width > self.size_approve or box_height > self.size_approve:
            return True
        elif box_area / image_area < self.min_area:
            print(
                f"Box area too small ({int(box_width)} x {int(box_height)}): {int(box_area)} / {image_area} = {(box_area/image_area):.2f} < {self.min_area}"
            )
            return False

        return True
