import json
import re
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image
from ultralytics.utils.ops import segment2box

from .detection_helpers import (
    index_detection,
    load_detections,
    load_detections_from_arrays,
    segmentation_to_box,
    txt_to_xywh,
    xywh_to_xyxy,
    xyxy_to_mask,
)
from .general import create_dataset_yaml, get_box_key, parallel
from .image_helpers import (
    add_bounding_box,
    add_bounding_boxes,
    add_label,
    add_mask,
    add_segmentation_masks,
    add_text,
    get_image_folders,
    resize_image,
)
from .pickle_helpers import load_pickle, save_pickle
from .plot import plot_images


def segmentation_to_xyxy(image: Image.Image, single_segmentation: list[float]):
    """Convert Segmentation floats to xyxy boxes."""
    w, h = image.size
    # Note: skip class_id, first index
    xyxy = segment2box(
        np.array(single_segmentation[1:]).reshape(-1, 2), width=w, height=h
    )
    return xyxy * [w, h, w, h]


def segmentation_to_mask(image: Image.Image, single_segmentation: list[float]):
    """Convert Segmentation floats to mask."""
    w, h = image.size
    # Note: skip class_id, first index
    mask = sv.polygon_to_mask(
        np.array(np.array(single_segmentation[1:]).reshape(-1, 2) * [w, h], dtype=int),
        (w, h),
    )
    return mask


def get_xywh_from_image_path(image_path: Path) -> list[float] | None:
    detect_path = image_path.parent.parent / "detect" / f"{image_path.stem}.txt"
    if not detect_path.exists():
        return None

    xywh = [
        list(map(float, line.split(" ")))[1:]
        for line in detect_path.read_text().split("\n")
        if line.strip()
    ]
    return xywh


def xyxy_to_xywh(image: Image.Image, xyxy: list[float]) -> list[float]:
    """Convert xyxy boxes to x_center, y_center, width, height."""
    w, h = image.size
    x_center = (xyxy[0] + xyxy[2]) / 2 / w
    y_center = (xyxy[1] + xyxy[3]) / 2 / h
    width = (xyxy[2] - xyxy[0]) / w
    height = (xyxy[3] - xyxy[1]) / h
    return [x_center, y_center, width, height]


def get_detection(xyxy: list[list[float]], mask: list[list[float]]):
    return sv.Detections(
        xyxy=np.array(xyxy).reshape(-1, 4),
        mask=np.array(mask).astype(bool),
        class_id=np.array([0] * len(xyxy)),
    )


def parse_response(response: str) -> list[dict[str, str | float]]:
    # Regular expression to match JSON objects or arrays in the text
    json_match = re.search(r"(\{.*?\}|\[.*?\])", response, re.DOTALL)

    if json_match:
        cleaned_response = json_match.group(1)  # Extract the matched JSON part
        try:
            # Parse the cleaned response as JSON
            parsed_json = json.loads(cleaned_response)
            return [parsed_json] if isinstance(parsed_json, dict) else parsed_json
        except json.JSONDecodeError:
            raise ValueError(f"Found JSON but couldn't parse it: {cleaned_response}")
    else:
        raise ValueError(f"No JSON found in the response: {response}")


def resize_to_min_dimension(image: Image.Image, min_size: int):
    """Resize the image so that the smallest dimension is at least `min_size`.

    This function preserves the aspect ratio of the image.

    Args:
        image (PIL.Image.Image): The image to resize.
        min_size (int): The minimum dimension to resize to.

    Returns:
        PIL.Image.Image: The resized image.
    """
    width, height = image.size
    scale_factor = min_size / min(width, height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
