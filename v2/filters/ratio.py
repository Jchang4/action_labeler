import supervision as sv
from PIL import Image

from action_labeler.v2.base import IFilter


class SmallDetectionsFilter(IFilter):
    """Filter out detections that are too small relative to the image size."""

    min_area: float

    def __init__(self, min_area: float = 0.05):
        self.min_area = min_area

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

        return float(box_area) / float(image_area) >= self.min_area


class ApproveLargeDetectionsFilter(IFilter):
    """Approve detections that are a minimum size."""

    min_pixels: int

    def __init__(self, min_pixels: int = 300):
        self.min_pixels = min_pixels

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

        if box_width >= self.min_pixels and box_height >= self.min_pixels:
            return True

        return False
