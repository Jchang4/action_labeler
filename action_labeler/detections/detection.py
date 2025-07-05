from pathlib import Path

import numpy as np

from action_labeler.helpers.detections_helpers import (
    segmentation_points_to_xywh,
    xywh_to_segmentation_points,
    xywhs_to_xyxys,
    xyxys_to_xywhs,
)
from action_labeler.helpers.yolov8_dataset import yolov8_labels_to_row


class Detection:
    xyxy: np.ndarray
    segmentation_points: list[list[float]]
    class_id: np.ndarray
    image_size: tuple[int, int]  # as (width, height)

    def __init__(
        self,
        xyxy: np.ndarray,
        segmentation_points: list[list[float]],
        class_id: np.ndarray,
        image_size: tuple[int, int],
    ):
        # Check the shapes of the inputs
        assert (
            len(xyxy) == len(segmentation_points) == len(class_id)
        ), f"The number of detections must match the number of masks and class ids. Got {len(xyxy)} {len(segmentation_points)} {len(class_id)}"
        assert xyxy.shape == (len(xyxy), 4)
        assert class_id.shape == (len(class_id),)
        assert image_size[0] > 0 and image_size[1] > 0

        self.xyxy = xyxy
        self.segmentation_points = segmentation_points
        self.class_id = class_id
        self.image_size = image_size

    @classmethod
    def from_text_path(
        cls, text_path: Path | str, image_size: tuple[int, int]
    ) -> "Detection":
        # Determine if the text path is a detection or segmentation text path by checking the number of numbers in the first line
        rows = yolov8_labels_to_row(text_path)
        if len(rows[0]) == 5:
            return cls.from_detection_text_path(text_path, image_size)
        elif len(rows[0]) > 5:
            return cls.from_segmentation_text_path(text_path, image_size)
        else:
            raise ValueError(
                f"Invalid number of numbers in the first line of {text_path}"
            )

    @classmethod
    def from_detection_text_path(
        cls, text_path: Path | str, image_size: tuple[int, int]
    ) -> "Detection":
        rows = yolov8_labels_to_row(text_path)
        class_ids = [row[0] for row in rows]
        xywhs = [row[1:] for row in rows]
        xyxys = xywhs_to_xyxys(xywhs, image_size)
        segmentation_points = [xywh_to_segmentation_points(xywh) for xywh in xywhs]
        return cls(
            xyxy=np.array(xyxys).reshape(-1, 4),
            segmentation_points=segmentation_points,
            class_id=np.array(class_ids),
            image_size=image_size,
        )

    @classmethod
    def from_segmentation_text_path(
        cls, text_path: Path | str, image_size: tuple[int, int]
    ) -> "Detection":
        rows = yolov8_labels_to_row(text_path)
        class_ids = [row[0] for row in rows]
        segmentation_points = [row[1:] for row in rows]

        xywhs = [
            segmentation_points_to_xywh(segmentation_point)
            for segmentation_point in segmentation_points
        ]
        xyxys = xywhs_to_xyxys(xywhs, image_size)

        return cls(
            xyxy=np.array(xyxys).reshape(-1, 4),
            segmentation_points=segmentation_points,
            class_id=np.array(class_ids),
            image_size=image_size,
        )

    @classmethod
    def empty(cls, image_size: tuple[int, int] = (0, 0)) -> "Detection":
        return cls(
            xyxy=np.array([]).reshape(-1, 4),
            segmentation_points=[],
            class_id=np.array([]),
            image_size=image_size,
        )

    def get_index(self, index: int) -> "Detection":
        return self.__class__(
            xyxy=np.array([self.xyxy[index]]).reshape(1, 4),
            segmentation_points=[self.segmentation_points[index]],
            class_id=np.array([self.class_id[index]]),
            image_size=self.image_size,
        )

    def copy(self) -> "Detection":
        return self.__class__(
            xyxy=self.xyxy.copy(),
            segmentation_points=self.segmentation_points.copy(),
            class_id=self.class_id.copy(),
            image_size=self.image_size,
        )

    def is_empty(self) -> bool:
        return len(self.xyxy) == 0

    @property
    def xywh(self) -> list[tuple[float, float, float, float]]:
        return xyxys_to_xywhs(self.xyxy, self.image_size)

    def __str__(self):
        return (
            f"<Detection num_detections={len(self.xyxy)} image_size={self.image_size}>"
        )

    def __repr__(self):
        return self.__str__()
