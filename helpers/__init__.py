import json
import pickle
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import supervision as sv
import yaml
from PIL import Image, ImageStat
from tqdm.auto import tqdm
from ultralytics.utils.ops import segment2box

from .general import parallel
from .image_helpers import add_bounding_box, add_label, add_mask, resize_image


def load_pickle(path: str | Path, filename: str = "classification.pickle"):
    path = Path(path)

    if not (path / filename).exists():
        return defaultdict(dict)

    with open(path / filename, "rb") as f:
        return defaultdict(dict, pickle.load(f))


def save_pickle(data, path: str | Path, filename: str = "classification.pickle"):
    path = Path(path)

    print(f"Saving {len(data)} images to {str(path / filename)}")

    with open(path / filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved classification file.")


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


def xyxy_to_mask(
    image: Image.Image, xyxy: list[float], buffer_px: int = 4
) -> np.ndarray:
    """Convert xyxy boxes to mask."""
    width, height = image.size
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = (
        max(0, x1 - buffer_px),
        max(0, y1 - buffer_px),
        min(width, x2 + buffer_px),
        min(height, y2 + buffer_px),
    )

    mask[int(y1) : int(y2), int(x1) : int(x2)] = True
    return mask


def xywh_to_xyxy(image: Image.Image, xywh: list[float]) -> list[float]:
    """Convert x_center, y_center, width, height to xyxy boxes."""
    w, h = image.size
    x_center, y_center, width, height = xywh
    x1 = (x_center - width / 2) * w
    y1 = (y_center - height / 2) * h
    x2 = (x_center + width / 2) * w
    y2 = (y_center + height / 2) * h
    return [x1, y1, x2, y2]


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


def create_dataset_yaml(path: Path, classes: list[str]):
    data = {
        "train": "train/images",
        "val": "valid/images",
        "path": path.name,
        "nc": len(classes),
        "names": classes,
    }
    yaml.dump(data, open(path / "data.yaml", "w"))


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
