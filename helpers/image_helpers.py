import supervision as sv
from PIL import Image


def add_bounding_box(
    image: Image.Image,
    detections: sv.Detections,
    thickness: int = 4,
    color: sv.Color | sv.ColorPalette = sv.ColorPalette.DEFAULT,
) -> Image.Image:
    return sv.BoxAnnotator(color=color, thickness=thickness).annotate(
        scene=image, detections=detections
    )


def add_mask(
    image: Image.Image,
    detections: sv.Detections,
    opacity: float = 0.5,
    color: sv.Color | sv.ColorPalette = sv.ColorPalette.DEFAULT,
) -> Image.Image:
    return sv.MaskAnnotator(color=color, opacity=opacity).annotate(
        scene=image, detections=detections
    )


def add_label(
    image: Image.Image,
    detections: sv.Detections,
    color: sv.Color | sv.ColorPalette = sv.ColorPalette.DEFAULT,
    text_color: sv.Color | sv.ColorPalette = sv.Color.WHITE,
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 10,
    text_position: sv.Position = sv.Position.TOP_LEFT,
    color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
    border_radius: int = 0,
    smart_position: bool = False,
) -> Image.Image:
    return sv.LabelAnnotator(
        color=color,
        text_color=text_color,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
        text_position=text_position,
        color_lookup=color_lookup,
        border_radius=border_radius,
        smart_position=smart_position,
    ).annotate(scene=image, detections=detections)


def resize_image(image: Image.Image, size: int) -> Image.Image:
    """Resize the image while preserving aspect ratio.

    Args:
        image (PIL.Image.Image): The image to resize.
        size (int): The size to resize to. The larger dimension will be equal to `size`.

    Returns:
        PIL.Image.Image: The resized image.
    """
    width, height = image.size
    scale_factor = size / max(width, height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
