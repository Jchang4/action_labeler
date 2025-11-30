"""Preprocessors for batch mode that add numbered labels to all detections."""

from PIL.Image import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import add_bounding_box, add_text
from action_labeler.preprocessors.base import IPreprocessor


class AllTextPreprocessor(IPreprocessor):
    """Add numbered text labels to ALL detections in the image.

    This is designed for batch processing where the model sees all detections
    at once and needs to identify them by number.

    The numbering starts from 0 and corresponds to the detection index
    in the Detection object.

    Args:
        text_template: Template for text (use {index} as placeholder)
        text_color: RGB color tuple for text
        font_size: Font size for text

    Example:
        # Add "0", "1", "2", ... to each detection
        preprocessor = AllTextPreprocessor()

        # Add "Person 0", "Person 1", ... to each detection
        preprocessor = AllTextPreprocessor(text_template="Person {index}")
    """

    text_template: str = "{index}"
    text_color: tuple[int, int, int] = (255, 0, 0)
    font_size: int = 30

    def __init__(
        self,
        text_template: str = "{index}",
        text_color: tuple[int, int, int] = (255, 0, 0),
        font_size: int = 30,
    ):
        """Initialize the preprocessor.

        Args:
            text_template: Template for text (use {index} as placeholder)
            text_color: RGB color tuple for text
            font_size: Font size for text
        """
        self.text_template = text_template
        self.text_color = text_color
        self.font_size = font_size

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detection,
    ) -> Image:
        """Add numbered text to all detections.

        Note: The 'index' parameter is ignored since we process all detections.
        This is called with index=0 or None in batch mode.

        Args:
            image: The image to add text to
            index: Ignored in batch mode
            detections: All detections in the image

        Returns:
            Image with numbered text on all detections
        """
        # Add text to each detection
        for i in range(len(detections.xyxy)):
            text = self.text_template.format(index=i)
            image = add_text(
                image,
                i,
                detections,
                text,
                text_color=self.text_color,
                font_size=self.font_size,
            )

        return image


class AllNumberedBoundingBoxPreprocessor(IPreprocessor):
    """Add numbered bounding boxes to all detections.

    Combines bounding boxes with numbered text labels for batch processing.

    Args:
        text_template: Template for text (use {index} as placeholder)
        box_color: RGB color tuple for bounding box
        text_color: RGB color tuple for text
        box_width: Width of bounding box line
        font_size: Font size for text
    """

    text_template: str = "{index}"
    box_color: tuple[int, int, int] = (255, 0, 0)
    text_color: tuple[int, int, int] = (255, 255, 255)
    box_width: int = 3
    font_size: int = 30
    buffer_px: int = 0

    def __init__(
        self,
        text_template: str = "{index}",
        box_color: tuple[int, int, int] = (255, 0, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        box_width: int = 3,
        font_size: int = 30,
        buffer_px: int = 0,
    ):
        """Initialize the preprocessor.

        Args:
            text_template: Template for text (use {index} as placeholder)
            box_color: RGB color tuple for bounding box
            text_color: RGB color tuple for text
            box_width: Width of bounding box line
            font_size: Font size for text
            buffer_px: Buffer pixels around bounding box
        """
        self.text_template = text_template
        self.box_color = box_color
        self.text_color = text_color
        self.box_width = box_width
        self.font_size = font_size
        self.buffer_px = buffer_px

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: Detection,
    ) -> Image:
        """Add numbered bounding boxes and text to all detections.

        Args:
            image: The image to process
            index: Ignored in batch mode
            detections: All detections in the image

        Returns:
            Image with numbered bounding boxes on all detections
        """

        # Add bounding box and text to each detection
        for i in range(len(detections.xyxy)):
            # Add bounding box
            image = add_bounding_box(
                image,
                i,
                detections,
                color=self.box_color,
                width=self.box_width,
                buffer_px=self.buffer_px,
            )

            # Add numbered text
            text = self.text_template.format(index=i)
            image = add_text(
                image,
                i,
                detections,
                text,
                text_color=self.text_color,
                font_size=self.font_size,
            )

        return image
