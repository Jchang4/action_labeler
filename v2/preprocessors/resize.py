from PIL.Image import Image
from supervision.detection.core import Detections

from action_labeler.helpers import resize_image
from action_labeler.v2.base import IPreprocessor


class ResizePreprocessor(IPreprocessor):
    """Resize the image to the given size.
    Resize preserves aspect ratio.
    """

    size: int

    def __init__(self, size: int = 1080):
        self.size = size

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detections,
    ) -> Image:
        return resize_image(image, self.size)
