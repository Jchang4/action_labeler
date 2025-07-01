from PIL.Image import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import add_text
from action_labeler.preprocessors.base import IPreprocessor


class TextPreprocessor(IPreprocessor):
    """Add text to the image.

    Args:
        text (str): The text to add to the image.
    """

    text: str | None = None
    text_color: tuple[int, int, int] = (255, 0, 0)
    font_size: int = 20

    def __init__(
        self,
        text: str | None = None,
        text_color: tuple[int, int, int] = (255, 0, 0),
        font_size: int = 20,
    ):
        self.text = text
        self.text_color = text_color
        self.font_size = font_size

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detection,
    ) -> Image:
        text = self.text if self.text else str(index)
        return add_text(
            image,
            index,
            detections,
            text,
            text_color=self.text_color,
            font_size=self.font_size,
        )
