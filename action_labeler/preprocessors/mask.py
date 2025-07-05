from PIL.Image import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import add_mask
from action_labeler.preprocessors.base import IPreprocessor


class MaskPreprocessor(IPreprocessor):
    """Apply mask for current detection."""

    opacity: float
    color: str
    fill: bool

    def __init__(
        self,
        opacity: float = 0.5,
        color: str = "red",
        fill: bool = True,
    ):
        self.opacity = opacity
        self.color = color
        self.fill = fill

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detection,
    ) -> Image:
        return add_mask(
            image,
            index,
            detections,
            opacity=self.opacity,
            color=self.color,
            fill=self.fill,
        )
