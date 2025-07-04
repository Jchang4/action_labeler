from PIL.Image import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import add_bounding_box
from action_labeler.preprocessors.base import IPreprocessor


class BoundingBoxPreprocessor(IPreprocessor):
    """Add bounding box to the image.

    Args:
        buffer_px (int): The number of pixels to add to the bounding box.
        force_crop (bool): If True, crop the image to the bounding box even if there is only one detection.
    """

    buffer_px: int
    box_width: int
    color: tuple[int, int, int]

    def __init__(
        self,
        buffer_px: int = 4,
        box_width: int = 2,
        color: tuple[int, int, int] = (255, 0, 0),
    ):
        self.buffer_px = buffer_px
        self.box_width = box_width
        self.color = color

    def preprocess(self, image: Image, index: int, detections: Detection) -> Image:
        return add_bounding_box(
            image,
            index,
            detections,
            color=self.color,
            width=self.box_width,
            buffer_px=self.buffer_px,
        )


class AllBoundingBoxesPreprocessor(BoundingBoxPreprocessor):
    """Add bounding boxes to all detections in the image."""

    def preprocess(self, image: Image, index: int, detections: Detection) -> Image:
        for i in range(len(detections.xyxy)):
            image = super().preprocess(image, i, detections)
        return image
