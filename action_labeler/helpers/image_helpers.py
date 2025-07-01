import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from action_labeler.detections.detection import Detection


def add_bounding_box(
    image: Image.Image,
    index: int,
    detections: Detection,
    color: str = "red",
    width: int = 2,
) -> Image.Image:
    image = image.copy()

    draw = ImageDraw.Draw(image)
    xyxy = detections.xyxy[index]
    draw.rectangle(xyxy, outline=color, width=width)
    return image


def add_bounding_boxes(
    image: Image.Image,
    detections: Detection,
    color: str = "red",
    width: int = 2,
) -> Image.Image:
    for index in range(len(detections.xyxy)):
        image = add_bounding_box(image, index, detections, color, width)
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


def add_mask(
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


def add_masks(
    image: Image.Image,
    detections: Detection,
    color: tuple[int, int, int] = (255, 0, 0),
    opacity: float = 0.5,
) -> Image.Image:
    for index in range(len(detections.xyxy)):
        image = add_mask(image, index, detections, color, opacity)
    return image


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
