import json
import numpy as np
from pathlib import Path
import supervision as sv
from ultralytics.utils.ops import segment2box
import pickle
from PIL import Image, ImageStat
from collections import defaultdict
import yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from typing import Callable


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


def get_detection(xyxy: list[float], mask: list[float]):
    return sv.Detections(
        xyxy=np.array(xyxy),
        mask=np.array(mask).astype(bool),
        class_id=np.array([0] * len(xyxy)),
    )


def parse_response(response: str) -> list[dict[str, str | float]]:
    # Remove the triple backticks and 'json' string from the response
    if "```json" in response:
        cleaned_response = response.split("```json")[1].split("```")[0]
    else:
        cleaned_response = response

    # Parse the cleaned response as JSON
    parsed_json = json.loads(cleaned_response)

    return [parsed_json]


def create_dataset_yaml(path: Path, classes: list[str]):
    data = {
        "train": "train/images",
        "val": "valid/images",
        "path": path.name,
        "nc": len(classes),
        "names": classes,
    }
    yaml.dump(data, open(path / "data.yaml", "w"))


def get_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)

    # Convert the image to grayscale to calculate brightness
    gray_image = image.convert("L")  # Convert to grayscale

    # Calculate the average brightness
    stat = ImageStat.Stat(gray_image)
    average_brightness = stat.mean[0]  # Get the average value

    # Define background color based on brightness (threshold can be adjusted)
    bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)

    # Create a new image with the same size as the original, filled with the background color
    new_image = Image.new("RGB", image.size, bg_color)

    # Paste the original image on top of the background (use image as a mask if needed)
    new_image.paste(image, (0, 0), image if image.mode == "RGBA" else None)

    return new_image


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
