from PIL.Image import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import add_background_mask, add_mask
from action_labeler.preprocessors.base import IPreprocessor


class MaskPreprocessor(IPreprocessor):
    """Apply mask for current detection."""

    def __init__(self, opacity: float = 0.5, color: tuple[int, int, int] = (255, 0, 0)):
        self.opacity = opacity
        self.color = color

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detection,
    ) -> Image:
        return add_mask(
            image, index, detections, opacity=self.opacity, color=self.color
        )


class BackgroundMaskPreprocessor(IPreprocessor):
    """Apply mask for current detection."""

    def __init__(self, opacity: float = 0.5, color: tuple[int, int, int] = (255, 0, 0)):
        self.opacity = opacity
        self.color = color

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detection,
    ) -> Image:
        return add_background_mask(
            image, index, detections, opacity=self.opacity, color=self.color
        )
