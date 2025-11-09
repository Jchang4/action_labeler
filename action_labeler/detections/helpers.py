"""Helper functions for detection coordinate conversions.

All coordinate conversions for the Detection class.
Coordinates in YOLO format are normalized (0-1).
"""

from enum import Enum
from pathlib import Path

import numpy as np


class DetectionFormat(Enum):
    """Enum for YOLO detection formats."""

    BBOX = "bbox"
    SEGMENT = "segment"
    POSE = "pose"


def yolov8_labels_to_rows(label_path: Path | str) -> list[list[float]]:
    """Parse YOLO format label file into rows of float values.

    Args:
        label_path: Path to .txt file with YOLO labels

    Returns:
        List of rows, where each row is [class_id, ...values]
    """
    label_path = Path(label_path)
    rows = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and len(line.split()) > 1:
            rows.append([float(num) for num in line.split()])

    return rows


def detect_format(
    rows: list[list[float]], num_keypoints: int | None = None
) -> DetectionFormat:
    """Determine the detection format based on the number of values in rows.

    YOLO format detection rules:
    - BBOX: Exactly 5 values (class_id, x_center, y_center, width, height)
    - SEGMENT: More than 5 values (variable polygon points)
    - POSE: 5 bbox values + 2*num_keypoints (when num_keypoints is provided)

    Args:
        rows: List of detection rows from YOLO format file
        num_keypoints: Expected number of keypoints (required for pose detection)

    Returns:
        DetectionFormat enum value (BBOX, SEGMENT, or POSE)

    Raises:
        ValueError: If format cannot be determined or is invalid

    Examples:
        >>> rows = [[0, 0.5, 0.5, 0.2, 0.3]]  # bbox format
        >>> detect_format(rows)
        <DetectionFormat.BBOX: 'bbox'>

        >>> rows = [[0, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]]  # segment
        >>> detect_format(rows)
        <DetectionFormat.SEGMENT: 'segment'>

        >>> rows = [[0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.3, 0.5, 0.4]]  # pose with 2 kp
        >>> detect_format(rows, num_keypoints=2)
        <DetectionFormat.POSE: 'pose'>
    """
    if not rows:
        return DetectionFormat.BBOX

    num_values = len(rows[0])

    # Validate minimum values
    if num_values < 5:
        raise ValueError(
            f"Invalid YOLO format: expected at least 5 values, got {num_values}"
        )

    # Check for bbox format
    if num_values == 5:
        return DetectionFormat.BBOX

    # Check for pose format (if num_keypoints provided)
    if num_keypoints is not None:
        expected_pose_values = 5 + 2 * num_keypoints
        if num_values == expected_pose_values:
            return DetectionFormat.POSE

    # Default to segmentation for everything else
    return DetectionFormat.SEGMENT


def xywh_to_xyxy(
    xywh: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert normalized xywh to pixel xyxy coordinates.

    Args:
        xywh: (x_center, y_center, width, height) normalized [0-1]
        image_size: (width, height) in pixels

    Returns:
        (x1, y1, x2, y2) in pixels
    """
    x_center, y_center, width, height = xywh
    image_width, image_height = image_size

    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height

    return x1, y1, x2, y2


def xywhs_to_xyxys(
    xywhs: list[tuple[float, float, float, float]], image_size: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    """Convert list of normalized xywh to pixel xyxy coordinates."""
    return [xywh_to_xyxy(xywh, image_size) for xywh in xywhs]


def xyxy_to_xywh(
    xyxy: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert pixel xyxy to normalized xywh coordinates.

    Args:
        xyxy: (x1, y1, x2, y2) in pixels
        image_size: (width, height) in pixels

    Returns:
        (x_center, y_center, width, height) normalized [0-1]
    """
    x1, y1, x2, y2 = xyxy
    image_width, image_height = image_size

    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return x_center, y_center, width, height


def xyxys_to_xywhs(
    xyxys: np.ndarray | list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    """Convert pixel xyxy to normalized xywh coordinates."""
    return [xyxy_to_xywh(tuple(xyxy), image_size) for xyxy in xyxys]


def xywh_to_segmentation_points(xywh: tuple[float, float, float, float]) -> list[float]:
    """Convert normalized xywh bbox to segmentation polygon (4 corners).

    Args:
        xywh: (x_center, y_center, width, height) normalized [0-1]

    Returns:
        [x1, y1, x2, y2, x3, y3, x4, y4] - 4 corners as flat list
    """
    x, y, w, h = xywh
    # Top-left, top-right, bottom-right, bottom-left
    return [
        x - w / 2,
        y - h / 2,  # Top-left
        x + w / 2,
        y - h / 2,  # Top-right
        x + w / 2,
        y + h / 2,  # Bottom-right
        x - w / 2,
        y + h / 2,  # Bottom-left
    ]


def segmentation_points_to_xywh(
    segmentation_points: list[float],
) -> tuple[float, float, float, float]:
    """Convert segmentation polygon to normalized xywh bbox.

    Args:
        segmentation_points: [x1, y1, x2, y2, ..., xn, yn] normalized [0-1]

    Returns:
        (x_center, y_center, width, height) normalized [0-1]
    """
    points = np.array(segmentation_points).reshape(-1, 2)
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def keypoints_to_numpy(keypoints_flat: list[float], num_keypoints: int) -> np.ndarray:
    """Convert flat keypoint list to numpy array (2D format).

    Args:
        keypoints_flat: [x1, y1, x2, y2, ..., xn, yn]
        num_keypoints: Number of keypoints

    Returns:
        Array of shape (num_keypoints, 2) with [x, y]
    """
    keypoints = np.array(keypoints_flat).reshape(num_keypoints, 2)
    return keypoints


def numpy_to_keypoints_flat(keypoints: np.ndarray) -> list[float]:
    """Convert numpy keypoints array to flat list (2D format).

    Args:
        keypoints: Array of shape (num_keypoints, 2) with [x, y]

    Returns:
        Flat list [x1, y1, x2, y2, ..., xn, yn]
    """
    return keypoints.flatten().tolist()
