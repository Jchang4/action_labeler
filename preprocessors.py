from PIL import Image
import supervision as sv

from .base import BaseImagePreprocessor
from .helpers import get_detection


class MaskImagePreprocessor(BaseImagePreprocessor):
    annotator: sv.MaskAnnotator

    def __init__(self, opacity: float = 0.5):
        self.annotator = sv.MaskAnnotator(opacity=opacity)

    def preprocess(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> Image.Image:
        if len(detections.xyxy) <= 1:
            return image

        single_detection = get_detection(
            xyxy=[detections.xyxy[index]],
            mask=[detections.mask[index]],
        )
        return self.annotator.annotate(
            scene=image,
            detections=single_detection,
        )


class BoxImagePreprocessor(BaseImagePreprocessor):
    annotator: sv.BoxAnnotator

    def __init__(self):
        self.annotator = sv.BoxAnnotator()

    def preprocess(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> Image.Image:
        # if len(detections.xyxy) <= 1:
        #     return image

        single_detection = get_detection(
            xyxy=[detections.xyxy[index]],
            mask=[detections.mask[index]],
        )
        return self.annotator.annotate(
            scene=image,
            detections=single_detection,
        )


class CropImagePreprocessor(BaseImagePreprocessor):
    buffer_pct: float
    force_crop: bool

    def __init__(self, buffer_pct: float = 0.05, force_crop: bool = False):
        self.buffer_pct = buffer_pct
        self.force_crop = force_crop

    def preprocess(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> Image.Image:
        if not self.force_crop and len(detections.xyxy) <= 1:
            return image

        # Add buffer to the crop
        box_width = detections.xyxy[index][2] - detections.xyxy[index][0]
        box_height = detections.xyxy[index][3] - detections.xyxy[index][1]
        x_buffer = self.buffer_pct * box_width
        y_buffer = self.buffer_pct * box_height

        return image.crop(
            (
                max(0, int(detections.xyxy[index][0] - x_buffer)),
                max(0, int(detections.xyxy[index][1] - y_buffer)),
                min(image.width, int(detections.xyxy[index][2] + x_buffer)),
                min(
                    image.height,
                    int(detections.xyxy[index][3] + y_buffer),
                ),
            )
        )


class ResizeImagePreprocessor(BaseImagePreprocessor):
    def preprocess(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> Image.Image:
        image.thumbnail((1080, 1080))
        return image


class MinResizeImagePreprocessor(BaseImagePreprocessor):
    min_size: int

    def __init__(self, min_size: int = 640):
        self.min_size = min_size

    def preprocess(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> Image.Image:
        if image.width < self.min_size or image.height < self.min_size:
            image.thumbnail((self.min_size, self.min_size))
        return image


class MaxImageSizePreprocessor(BaseImagePreprocessor):
    max_size: int

    def __init__(self, max_size: int = 1920):
        self.max_size = max_size

    def preprocess(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> Image.Image:
        if image.width > self.max_size or image.height > self.max_size:
            image.thumbnail((self.max_size, self.max_size))
        return image
