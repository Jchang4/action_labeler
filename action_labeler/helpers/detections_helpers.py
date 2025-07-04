import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    raise ImportError(
        "Ultralytics requires the ultralytics package. Please install it with `pip install ultralytics`."
    )


def image_to_txt_path(image_path: Path | str) -> Path:
    parent_path = image_path.parent.parent
    txt_file_name = image_path.with_suffix(".txt").name
    return parent_path / "detect" / txt_file_name


def ultralytics_labels_to_xywh(txt_path: Path | str) -> list[list[float]]:
    """Convert a Ultralytics labels txt file to a list of xywh boxes.

    Ultralytics labels formats: https://docs.ultralytics.com/modes/predict/#working-with-results

    Note: this method can also handle segmentation and keypoint detection.

    Args:
        txt_path (Path | str): The path to the txt file.

    Returns:
        list[list[float]]: A list of xywh boxes.
    """
    txt_path = Path(txt_path)
    return [
        [float(num) for num in line.split(" ")][1:]  # Skip the class id
        for line in txt_path.read_text().splitlines()
        if line and len(line.split(" ")) > 1
    ]


def xyxy_to_xywh(
    xyxy: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert a list of xyxy coordinates to a list of xywh coordinates.

    Args:
        xyxy (tuple[float, float, float, float]): The xyxy coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).

    Returns:
        tuple[float, float, float, float]: The xywh coordinates.
    """
    x1, y1, x2, y2 = map(float, xyxy)
    image_width, image_height = image_size

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return [x_center, y_center, width, height]


def xyxys_to_xywhs(
    xyxys: list[tuple[float, float, float, float]], image_size: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    """Convert a list of xyxy coordinates to a list of xywh coordinates.

    Args:
        xyxys (list[tuple[float, float, float, float]]): The list of xyxy coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).

    Returns:
        list[tuple[float, float, float, float]]: The xywh coordinates.
    """
    return [xyxy_to_xywh(xyxy, image_size) for xyxy in xyxys]


def xywh_to_xyxy(
    xywh: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert a list of xywh coordinates to a list of xyxy coordinates.

    Args:
        xywh (tuple[float, float, float, float]): The xywh coordinates as (x_center, y_center, width, height) in normalized coordinates.
        image_size (tuple[int, int]): The size of the image.

    Returns:
        list[int]: The xyxy coordinates.
    """
    x_center, y_center, width, height = map(float, xywh)
    image_width, image_height = image_size

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    x1 *= image_width
    y1 *= image_height
    x2 *= image_width
    y2 *= image_height

    return x1, y1, x2, y2


def xywhs_to_xyxys(
    xywhs: list[tuple[float, float, float, float]], image_size: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    """Convert a list of xywh coordinates to a list of xyxy coordinates.

    Args:
        xywhs (list[tuple[float, float, float, float]]): The list of xywh coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).

    Returns:
        list[tuple[float, float, float, float]]: The xyxy coordinates.
    """
    return [xywh_to_xyxy(xywh, image_size) for xywh in xywhs]


def xyxy_to_mask(
    xyxy: list[float],
    image_size: tuple[int, int],
    buffer_px: int = 0,
) -> np.ndarray:
    """Convert xyxy boxes to mask."""
    width, height = image_size
    mask = np.zeros((width, height), dtype=bool)
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = (
        max(0, x1 - buffer_px),
        max(0, y1 - buffer_px),
        min(width, x2 + buffer_px),
        min(height, y2 + buffer_px),
    )

    mask[int(x1) : int(x2), int(y1) : int(y2)] = True
    return mask


def xyxys_to_masks(
    xyxys: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
    buffer_px: int = 0,
) -> list[list[bool]]:
    """Convert a list of xyxy coordinates to a list of masks.

    Args:
        xyxys (list[tuple[float, float, float, float]]): The list of xyxy coordinates.
        image_size (tuple[int, int]): The size of the image as (width, height).
        buffer_px (int, optional): The buffer in pixels. Defaults to 0.

    Returns:
        list[list[bool]]: The list of masks.
    """
    return [xyxy_to_mask(xyxy, image_size, buffer_px) for xyxy in xyxys]
