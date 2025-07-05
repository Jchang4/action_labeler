from pathlib import Path

import numpy as np


def image_to_txt_path(image_path: Path | str, detection_type: str = "detect") -> Path:
    """Get the path to the txt file for a given image path.

    Args:
        image_path (Path | str): The path to the image.
        detection_type (str, optional): The type of detection: "detect" or "segment". Defaults to "detect".

    Returns:
        Path: The path to the txt file.
    """
    image_path = Path(image_path)
    parent_path = image_path.parent.parent
    txt_file_name = image_path.with_suffix(".txt").name
    return parent_path / detection_type / txt_file_name


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


def xywh_to_segmentation_points(xywh: tuple[float, float, float, float]) -> list[float]:
    """Convert xywh (x_center, y_center, width, height) to segmentation points (x1, y1, x2, y2, ..., xn, yn).

    The segmentation points are the top left, top right, bottom right, and bottom left corners of the bounding box.

    Args:
        xywh (list[float]): The xywh coordinates as (x_center, y_center, width, height) in normalized coordinates.

    Returns:
        list[float]: The segmentation points.
    """
    x, y, w, h = xywh
    # top left, top right, bottom right, bottom left
    top_left_x = x - w / 2
    top_left_y = y - h / 2
    top_right_x = x + w / 2
    top_right_y = y - h / 2
    bottom_right_x = x + w / 2
    bottom_right_y = y + h / 2
    bottom_left_x = x - w / 2
    bottom_left_y = y + h / 2
    return [
        top_left_x,
        top_left_y,
        top_right_x,
        top_right_y,
        bottom_right_x,
        bottom_right_y,
        bottom_left_x,
        bottom_left_y,
    ]


def segmentation_points_to_xywh(
    segmentation_points: list[float],
) -> tuple[float, float, float, float]:
    """Convert segmentation points (x1, y1, x2, y2, ..., xn, yn) to xywh (x_center, y_center, width, height).

    Args:
        segmentation_points (list[float]): The segmentation points.

    Returns:
        tuple[float, float, float, float]: The xywh coordinates.
    """
    points_arr = np.array(segmentation_points).reshape(-1, 2)
    x_min = points_arr[:, 0].min()
    y_min = points_arr[:, 1].min()
    x_max = points_arr[:, 0].max()
    y_max = points_arr[:, 1].max()
    width = x_max - x_min
    height = y_max - y_min
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center, width, height
