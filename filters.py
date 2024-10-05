from PIL import Image
import supervision as sv

from .base import BaseImageFilter


class SmallDetectionsFilter(BaseImageFilter):
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
        if box_area / image_area < 0.2:
            return False
        elif box_width < 75 or box_height < 75:
            return False

        return True
