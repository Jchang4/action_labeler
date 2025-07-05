from pathlib import Path

import numpy as np

try:
    from ultralytics.engine.results import Results
except ImportError:
    raise ImportError(
        "Ultralytics requires the ultralytics package. Please install it with `pip install ultralytics`."
    )

from action_labeler.helpers.detections_helpers import (
    xywhs_to_xyxys,
    xyxys_to_masks,
    xyxys_to_xywhs,
)
from action_labeler.helpers.yolov8_dataset import ultralytics_labels_to_xywh


class Detection:
    xyxy: np.ndarray
    mask: np.ndarray
    class_id: np.ndarray
    image_size: tuple[int, int]  # as (width, height)

    def __init__(
        self,
        xyxy: np.ndarray,
        mask: np.ndarray,
        class_id: np.ndarray,
        image_size: tuple[int, int],
    ):
        self.xyxy = xyxy
        self.mask = mask
        self.class_id = class_id
        self.image_size = image_size

    @classmethod
    def from_ultralytics(cls, results: Results) -> "Detection":
        xyxy = results.boxes.xyxy.cpu().numpy()
        mask = (
            results.masks.xy.cpu().numpy()
            if results.masks
            else xyxys_to_masks(xyxy, results.orig_shape[::-1])
        )
        class_id = results.boxes.cls.cpu().numpy()
        # Ultralytics returns image size as (height, width)
        image_size = results.orig_shape[::-1]

        return cls(
            xyxy=xyxy,
            mask=mask,
            class_id=class_id,
            image_size=image_size,
        )

    @classmethod
    def from_text_path(
        cls, text_path: Path | str, image_size: tuple[int, int]
    ) -> "Detection":
        xywhs = ultralytics_labels_to_xywh(text_path)
        xyxys = xywhs_to_xyxys(xywhs, image_size)
        masks = xyxys_to_masks(xyxys, image_size)
        return cls(
            xyxy=xyxys,
            mask=masks,
            class_id=np.zeros(len(xyxys)),
            image_size=image_size,
        )

    @classmethod
    def empty(cls, image_size: tuple[int, int] = (0, 0)) -> "Detection":
        return cls(
            xyxy=np.array([]),
            mask=np.array([]),
            class_id=np.array([]),
            image_size=image_size,
        )

    def copy(self) -> "Detection":
        return self.__class__(
            xyxy=self.xyxy.copy(),
            mask=self.mask.copy(),
            class_id=self.class_id.copy(),
            image_size=self.image_size,
        )

    def is_empty(self) -> bool:
        return len(self.xyxy) == 0

    @property
    def xywhn(self) -> np.ndarray:
        return np.array(xyxys_to_xywhs(self.xyxy, self.image_size))

    def __str__(self):
        return (
            f"<Detection num_detections={len(self.xyxy)} image_size={self.image_size}>"
        )

    def __repr__(self):
        return self.__str__()
