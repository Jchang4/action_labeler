from PIL.Image import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import crop_image
from action_labeler.preprocessors.base import IPreprocessor


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
        detections: Detection,
    ) -> Image:
        if not self.force_crop and len(detections.xyxy) <= 1:
            return image

        xyxy = detections.xyxy[index]

        # Add buffer to the crop
        box_width = xyxy[2] - xyxy[0]
        box_height = xyxy[3] - xyxy[1]
        x_buffer = self.buffer_pct * box_width
        y_buffer = self.buffer_pct * box_height
        buffer_px = int(max(x_buffer, y_buffer))

        return crop_image(image, index, detections, buffer_px=buffer_px)
