from pathlib import Path

import supervision as sv
from PIL import Image, ImageDraw


def get_image_folders(root_dir: Path, exclude_filters: list[str] = []) -> list[Path]:
    image_folders = []

    for path in root_dir.rglob("*"):
        if (
            path.is_dir()
            and path.name == "images"
            and not any([f in str(path) for f in exclude_filters])
        ):
            image_folders.append(path.parent)
    return sorted(image_folders, key=lambda x: (len(str(x).split("/")), str(x)))


#######################################
#### Supervision Annotators ###########
#######################################
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


#######################################
#### PIL Annotators ##################
#######################################
def add_bounding_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    outline_color: str | tuple = "red",
    width: int = 2,
    buffer_px: int = 0,
):
    """
    Draw bounding boxes on an image.

    Args:
        image (PIL.Image.Image): The original image.
        boxes (list[list[float]]): A list of bounding boxes, each defined by [x_center, y_center, width, height]
                                   with normalized values (0 to 1).
        outline_color (str or tuple): Color for the bounding box outline. Default is 'red'.
        width (int): Width of the bounding box outline. Default is 2.

    Returns:
        PIL.Image.Image: The image with bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    for box in boxes:
        x_center, y_center, box_width, box_height = box
        # Convert normalized coordinates to absolute pixel values
        x_center *= img_width
        y_center *= img_height
        box_width *= img_width
        box_height *= img_height

        # Calculate the top-left and bottom-right coordinates
        left = x_center - box_width / 2
        top = y_center - box_height / 2
        right = x_center + box_width / 2
        bottom = y_center + box_height / 2

        # Add buffer
        left -= buffer_px
        top -= buffer_px
        right += buffer_px
        bottom += buffer_px

        # Cap the coordinates at the image boundaries
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        # Draw the rectangle
        draw.rectangle([left, top, right, bottom], outline=outline_color, width=width)

    return image


def add_segmentation_masks(
    image: Image.Image,
    normalized_segments: list[list[float]],
    mask_color=(255, 0, 0, 128),
) -> Image.Image:
    """
    Apply segmentation masks to an image.

    Args:
        image (PIL.Image.Image): The original image.
        normalized_segments (list[list[float]]): A list of segmentation points, where each segmentation is a list of
                                                [x1, y1, x2, y2, ..., xn, yn] coordinates in normalized format (0 to 1).
        mask_color (tuple): RGBA color tuple for the mask with transparency (default is semi-transparent red).

    Returns:
        PIL.Image.Image: The image with segmentation masks applied.
    """
    # Ensure the image has an alpha channel
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create an overlay image for masks
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = image.size

    for points in normalized_segments:
        # Convert normalized coordinates to absolute pixel values
        polygon = [(x * width, y * height) for x, y in zip(points[::2], points[1::2])]
        # Draw the polygon on the overlay
        draw.polygon(polygon, fill=mask_color)

    # Composite the overlay with the original image
    combined = Image.alpha_composite(image, overlay)

    return combined
