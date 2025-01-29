from PIL.Image import Image
from supervision.detection.core import Detections

from action_labeler.v2.base import IPreprocessor


class CropPreprocessor(IPreprocessor):
    """Crop the image to the detection box.

    If force_crop is False, only crop if there is more than one detection.
    """

    buffer_pct: float
    force_crop: bool

    def __init__(self, buffer_pct: float = 0.05, force_crop: bool = False):
        self.buffer_pct = buffer_pct
        self.force_crop = force_crop

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detections,
    ) -> Image:
        if not self.force_crop and len(detections.xyxy) <= 1:
            return image

        xyxy = detections.xyxy[index]

        # Add buffer to the crop
        box_width = xyxy[2] - xyxy[0]
        box_height = xyxy[3] - xyxy[1]
        x_buffer = self.buffer_pct * box_width
        y_buffer = self.buffer_pct * box_height

        return image.crop(
            (
                max(0, int(xyxy[0] - x_buffer)),
                max(0, int(xyxy[1] - y_buffer)),
                min(image.width, int(xyxy[2] + x_buffer)),
                min(image.height, int(xyxy[3] + y_buffer)),
            )
        )
