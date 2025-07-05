import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from action_labeler.detections.detection import Detection


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


def add_bounding_box(
    image: Image.Image,
    index: int,
    detections: Detection,
    color: str = "red",
    width: int = 2,
    buffer_px: int = 0,
) -> Image.Image:
    image = image.copy()

    draw = ImageDraw.Draw(image)
    xyxy = detections.xyxy[index]
    xyxy = (
        max(0, xyxy[0] - buffer_px),
        max(0, xyxy[1] - buffer_px),
        min(image.width, xyxy[2] + buffer_px),
        min(image.height, xyxy[3] + buffer_px),
    )
    draw.rectangle(xyxy, outline=color, width=width)
    return image


def add_bounding_boxes(
    image: Image.Image,
    detections: Detection,
    color: str = "red",
    width: int = 2,
    buffer_px: int = 0,
) -> Image.Image:
    for index in range(len(detections.xyxy)):
        image = add_bounding_box(image, index, detections, color, width, buffer_px)
    return image


def crop_image(
    image: Image.Image,
    index: int,
    detections: Detection,
    buffer_px: int = 0,
) -> Image.Image:
    image = image.copy()

    xyxy = detections.xyxy[index]
    xyxy = (
        max(0, xyxy[0] - buffer_px),
        max(0, xyxy[1] - buffer_px),
        min(image.width, xyxy[2] + buffer_px),
        min(image.height, xyxy[3] + buffer_px),
    )
    return image.crop(xyxy)


def add_segmentation_masks(
    image: Image.Image,
    normalized_segments: list[list[float]],
    mask_color: str | list[str] = "red",
    opacity: float = 0.3,
    outline_width: int = 4,
    fill: bool = False,
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
        fill (bool): Whether to fill the polygon with the mask color. Default is False.

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
        if fill:
            draw.polygon(polygon, fill=color, outline=color, width=outline_width)
        else:
            draw.polygon(polygon, outline=color, width=outline_width)

    # Composite the overlay with the original image
    combined = Image.alpha_composite(image, overlay)

    return combined


def add_mask(
    image: Image.Image,
    index: int,
    detections: Detection,
    color: tuple[int, int, int] = (255, 0, 0),
    opacity: float = 0.5,
    fill: bool = False,
) -> Image.Image:
    image = image.copy()
    return add_segmentation_masks(
        image,
        detections.get_index(index).segmentation_points,
        mask_color=color,
        opacity=opacity,
        fill=fill,
    )


def add_masks(
    image: Image.Image,
    detections: Detection,
    color: tuple[int, int, int] = (255, 0, 0),
    opacity: float = 0.5,
    fill: bool = False,
) -> Image.Image:
    image = image.copy()
    return add_segmentation_masks(
        image,
        detections.segmentation_points,
        mask_color=color,
        opacity=opacity,
        fill=fill,
    )


def add_background_mask(
    image: Image.Image,
    index: int,
    detections: Detection,
    color: tuple[int, int, int] = (255, 0, 0),
    opacity: float = 0.5,
) -> Image.Image:
    image = image.copy()

    # Create a transparent overlay for the mask
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Get the mask for this detection
    mask = detections.mask[index]
    # Flip the mask so true = false and false = true
    mask = ~mask

    # Convert the image to RGBA if it's not already
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create a colored mask where mask is True
    mask_array = np.array(mask)
    # Transpose mask_array to match the expected dimensions
    # The mask is created with shape (width, height) but we need (height, width) for image coordinates
    mask_array = (
        mask_array.T
    )  # Transpose to convert from (width, height) to (height, width)
    y_indices, x_indices = np.where(mask_array)

    # Fill the mask area with the color
    for y, x in zip(y_indices, x_indices):
        draw.point((x, y), fill=(*color, int(255 * opacity)))

    # Composite the overlay onto the original image
    return Image.alpha_composite(image, overlay)


def add_text(
    image: Image.Image,
    index: int,
    detections: Detection,
    text: str,
    text_color: tuple[int, int, int] = (255, 0, 0),
    font_size: int = 20,
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
) -> Image.Image:
    image = image.copy()

    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)
    xyxy = detections.xyxy[index]
    x_min, y_min, x_max, y_max = xyxy

    # Calculate text dimensions
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position text at the top of the bounding box
    text_x = x_min
    text_y = max(
        0, y_min - text_height - 2
    )  # Place text above the box with small margin

    # Add background for better visibility
    draw.rectangle(
        [text_x, text_y, text_x + text_width, text_y + text_height], fill=(0, 0, 0, 128)
    )

    # Draw the text
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    return image


def add_texts(
    image: Image.Image,
    detections: Detection,
    texts: list[str],
    text_color: tuple[int, int, int] = (255, 0, 0),
    font_size: int = 20,
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
) -> Image.Image:
    assert len(texts) == len(
        detections.xyxy
    ), "Number of texts must match number of detections"

    for index, text in enumerate(texts):
        image = add_text(
            image, index, detections, text, text_color, font_size, font_path
        )
    return image
