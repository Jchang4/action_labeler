from pathlib import Path

import numpy as np
import supervision as sv


def txt_to_xywh(txt_path: Path | str) -> list[list[float]]:
    """Convert a txt file to a list of xywh boxes.

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


def xywh_to_xyxy(xywh: list[float], image_size: tuple[int, int]) -> list[float]:
    x_center = xywh[0] * image_size[0]
    y_center = xywh[1] * image_size[1]
    w = xywh[2] * image_size[0]
    h = xywh[3] * image_size[1]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return [x1, y1, x2, y2]


def xyxy_to_mask(
    xyxy: list[float],
    image_size: tuple[int, int],
    buffer_px: int = 0,
) -> np.ndarray:
    """Convert xyxy boxes to mask."""
    width, height = image_size
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


def segmentation_to_box(
    segmentation_points: list[float],
) -> tuple[float, float, float, float]:
    """
    Convert segmentation points to a normalized bounding box in xywh format.

    Args:
        segmentation_points (list[float]): A list of segmentation points in normalized coordinates
                                           [x1, y1, x2, y2, ..., xn, yn].

    Returns:
        tuple: A tuple representing the bounding box (x_center, y_center, width, height) in normalized coordinates.
    """
    # Extract x and y coordinates
    x_coords = segmentation_points[::2]
    y_coords = segmentation_points[1::2]

    # Calculate the minimum and maximum coordinates
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Compute the center coordinates, width, and height
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def load_detections(txt_path: Path | str, image_size: tuple[int, int]) -> sv.Detections:
    txt_path = Path(txt_path)
    xywhs = txt_to_xywh(txt_path)
    xyxys = [xywh_to_xyxy(xywh, image_size) for xywh in xywhs]
    masks = [xyxy_to_mask(xyxy, image_size) for xyxy in xyxys]
    return sv.Detections(
        xyxy=np.array(xyxys).reshape(-1, 4),
        mask=np.array(masks).reshape(-1, image_size[1], image_size[0]),
        class_id=np.array([0] * len(xyxys)),
    )


def load_detections_from_arrays(
    xyxys: list[list[float]],
    masks: list[list[float]],
    image_size: tuple[int, int],
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(xyxys).reshape(-1, 4),
        mask=np.array(masks).reshape(-1, image_size[1], image_size[0]),
        class_id=np.array([0] * len(xyxys)),
    )


def reverse_mask(detections: sv.Detections) -> sv.Detections:
    return sv.Detections(
        xyxy=detections.xyxy,
        mask=~detections.mask,
        class_id=detections.class_id,
    )


def index_detection(detections: sv.Detections, index: int) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([detections.xyxy[index]]),
        mask=np.array([detections.mask[index]]).astype(bool),
        class_id=np.array([detections.class_id[index]]),
    )
