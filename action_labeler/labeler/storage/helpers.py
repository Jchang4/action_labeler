from PIL import Image, ImageDraw, ImageFont

from action_labeler.action_labeler.helpers.detections_helpers import xywh_to_xyxy
from action_labeler.action_labeler.labeler.storage.metadata import LabeledDetection


def draw_bounding_box(
    image: Image.Image,
    detection: LabeledDetection,
    color: str = "red",
    width: int = 2,
    buffer_px: int = 0,
    show_label: bool = False,
) -> Image.Image:
    """Draw a bounding box on an image.

    Args:
        image: Image to draw on
        detection: LabeledDetection object

    Returns:
        Image with bounding box drawn on it
    """
    xywh = detection.xywh
    xywh = (
        max(0, xywh[0] - buffer_px),
        max(0, xywh[1] - buffer_px),
        min(image.width, xywh[2] + buffer_px),
        min(image.height, xywh[3] + buffer_px),
    )
    xyxy = xywh_to_xyxy(xywh, image.size)
    draw = ImageDraw.Draw(image)
    draw.rectangle(xyxy, outline=color, width=width)

    # Add label to center
    if show_label:
        # Font Size 20
        font = ImageFont.load_default(size=20)
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2
        draw.text((center_x, center_y), detection.label, fill=color, font=font)

    return image


def get_image_with_detections(
    detections: list[LabeledDetection],
    show_label: bool = False,
) -> Image.Image:
    """Get an image with detections overlaid on it.

    Args:
        detections: List of LabeledDetection objects

    Returns:
        Image with detections overlaid on it
    """
    # Ensure image_path is same for all detections
    image_path = detections[0].image_path
    for detection in detections:
        if detection.image_path != image_path:
            raise ValueError("Image paths must be the same for all detections")

    image = Image.open(image_path)
    for detection in detections:
        image = draw_bounding_box(image, detection, show_label=show_label)
    return image
