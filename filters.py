from PIL import Image
import supervision as sv

from .base import BaseImageFilter


class SmallDetectionsFilter(BaseImageFilter):
    """Filter out detections that are too small."""

    min_area: float
    min_size: int
    size_approve: int

    def __init__(
        self, min_area: float = 0.1, min_size: int = 75, size_approve: int = 300
    ):
        self.min_area = min_area
        self.min_size = min_size
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

        # For large images, set a minimum box size
        if box_width > self.size_approve and box_height > self.size_approve:
            return True
        elif box_area / image_area < self.min_area:
            # print(f"Box area too small: {box_area} / {image_area} < {self.min_area}")
            return False
        elif box_width < self.min_size or box_height < self.min_size:
            # print(
            #     f"Box size too small: {box_width} < {self.min_size} or {box_height} < {self.min_size}"
            # )
            return False

        return True
