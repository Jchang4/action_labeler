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
        xyxy=np.array(xyxy),
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
    # Get original dimensions
    width, height = image.size

    # Determine the scaling factor to ensure the smaller dimension reaches min_size
    if width < height:
        scale_factor = min_size / width
    else:
        scale_factor = min_size / height

    # Calculate the new size while preserving the aspect ratio
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def parallel(
    f: Callable,
    items: list,
    *args: list,
    n_workers=24,
    **kwargs,
):
    """Applies `func` in parallel to `items`, using `n_workers`

    Args:
        f (function): function to apply
        items (list): list of items to apply `f` to
        n_workers (int, optional): number of workers. Defaults to 24.

    Returns:
        list: list of results
    """
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        r = list(
            tqdm(
                ex.map(f, items, *args, **kwargs),
                total=len(items),
            )
        )
    if any([o is Exception for o in r]):
        raise Exception(r)
    return r
