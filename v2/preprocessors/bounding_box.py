import numpy as np
import supervision as sv
from PIL.Image import Image

from action_labeler.v2.base import IPreprocessor


class BoundingBoxPreprocessor(IPreprocessor):
    """Add bounding box to the image.

    Args:
        buffer_px (int): The number of pixels to add to the bounding box.
        force_crop (bool): If True, crop the image to the bounding box even if there is only one detection.
    """

    annotator: sv.BoxAnnotator
    buffer_px: int

    def __init__(self, buffer_px: int = 4):
        self.annotator = sv.BoxAnnotator()
        self.buffer_px = buffer_px

    def preprocess(self, image: Image, index: int, detections: sv.Detections) -> Image:
        return self.annotator.annotate(
            scene=image,
            detections=self.index_detection(detections, index),
        )

    def index_detection(self, detections: sv.Detections, index: int) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array([detections.xyxy[index]]),
            mask=np.array([detections.mask[index]]).astype(bool),
            class_id=np.array([0]),
        )
