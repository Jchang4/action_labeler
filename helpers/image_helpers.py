from pathlib import Path

import supervision as sv
from PIL import Image, ImageColor, ImageDraw, ImageFont


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
    mask_color: str | list[str] = "red",
    opacity: float = 0.3,
    outline_width: int = 4,
):
    """
    Apply segmentation masks to an image with adjustable opacity.

    Args:
        image (PIL.Image.Image): The original image.
        normalized_segments (list[list[float]]): A list of segmentation points, where each segmentation is a list of
                                                [x1, y1, x2, y2, ..., xn, yn] coordinates in normalized format (0 to 1).
        mask_color (str or tuple): Color for the mask. Can be a color name (e.g., 'red') or an RGB tuple (default is 'red').
        opacity (float): Opacity level of the mask, ranging from 0.0 (fully transparent) to 1.0 (fully opaque).
        outline_width (int): Width of the mask outline. Default is 4.

    Returns:
        PIL.Image.Image: The image with segmentation masks applied.
    """
    # Ensure the opacity is within the valid range
    if not (0.0 <= opacity <= 1.0):
        raise ValueError("Opacity must be between 0.0 and 1.0")

    # Convert color name to RGB tuple if necessary and add alpha value
    alpha = int(255 * opacity)
    if isinstance(mask_color, str):
        mask_color = [ImageColor.getrgb(mask_color) + (alpha,)] * len(
            normalized_segments
        )
    else:
        mask_color = [ImageColor.getrgb(color) + (alpha,) for color in mask_color]

    assert len(mask_color) >= len(
        normalized_segments
    ), "mask_color must have the same length as normalized_segments"

    # Ensure the image has an alpha channel
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create an overlay image for masks with the same size as the original image
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = image.size

    for points, color in zip(normalized_segments, mask_color):
        # Convert normalized coordinates to absolute pixel values
        polygon = [(x * width, y * height) for x, y in zip(points[::2], points[1::2])]
        # Draw the polygon on the overlay with the specified opacity
        draw.polygon(polygon, outline=color, width=outline_width)

    # Composite the overlay with the original image
    combined = Image.alpha_composite(image, overlay)

    return combined


def add_text(
    image: Image.Image,
    boxes: list[list[float]],
    texts: list[str] | None = None,
    text_position: str = "top",
    font_path: str | None = None,
    font_size: int = 20,
    text_color: str | tuple[int, int, int] = "red",
):
    """
    Add text labels to an image at specified bounding box locations.

    Args:
        image (PIL.Image.Image): The original image.
        boxes (list[list[float]]): A list of bounding boxes, each defined by [x_center, y_center, width, height]
                                   with normalized values (0 to 1).
        texts (list[str], optional): A list of text labels corresponding to each bounding box. If None, indices are used.
        text_position (str): Position of the text relative to the bounding box ('top', 'bottom', 'center'). Default is 'top'.
        font_path (str, optional): Path to the TrueType font file to use. If None, the default font is used.
        font_size (int): Size of the font. Default is 20.
        text_color (str or tuple): Color of the text. Default is 'red'.

    Returns:
        PIL.Image.Image: The image with text labels added.
    """
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    # Load the specified font or default to a basic font
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default(font_size)

    if isinstance(text_color, str):
        text_color = [text_color] * len(boxes)

    assert len(text_color) >= len(
        boxes
    ), "text_color must have the same length as boxes"

    if texts is None:
        # Generate default text labels as indices starting from 1
        texts = [str(i + 1) for i in range(len(boxes))]

    for box, text, color in zip(boxes, texts, text_color):
        x_center, y_center, box_width, box_height = box
        # Convert normalized coordinates to absolute pixel values
        x_center *= img_width
        y_center *= img_height
        box_width *= img_width
        box_height *= img_height

        # Calculate the top-left and bottom-right coordinates of the bounding box
        left = x_center - box_width / 2
        top = y_center - box_height / 2
        bottom = y_center + box_height / 2

        # Determine text size using textbbox
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = (text_bbox[3] - text_bbox[1]) * 1.5

        # Determine text position
        if text_position == "top":
            text_x = left
            text_y = top
            # text_y = text_height + text_height * 0.25  # Slightly above the bounding box
        elif text_position == "bottom":
            text_x = left
            text_y = bottom - text_height
        elif text_position == "center":
            text_x = left + box_width / 2
            text_y = top + box_height / 2
        else:
            raise ValueError("text_position must be 'top' or 'bottom'")

        # Draw text background for better visibility
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height],
            fill=(0, 0, 0, 128),  # Semi-transparent black background
        )

        # Draw the text
        draw.text((text_x, text_y), text, font=font, fill=color)

    return image
