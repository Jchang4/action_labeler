from PIL.Image import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import resize_image
from action_labeler.preprocessors.base import IPreprocessor


class ResizePreprocessor(IPreprocessor):
    """Resize the image to the given size.

    Args:
        size (int): The size to resize to. The larger dimension will be equal to `size`.
    """

    size: int

    def __init__(self, size: int = 1080):
        self.size = size

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detection,
    ) -> Image:
        return resize_image(image, self.size)
