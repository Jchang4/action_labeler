from pathlib import Path

import numpy as np
import supervision as sv

from action_labeler.helpers import xywh_to_xyxy, xyxy_to_mask
from action_labeler.v2.base import IVisionLanguageModel


def image_to_detect_path(image_path: Path) -> Path:
    folder = image_path.parent.parent / "detect"
    return folder / image_path.with_suffix(".txt").name


def image_to_detections(image_path: Path, model: IVisionLanguageModel) -> sv.Detections:
    txt_path = image_to_detect_path(image_path)
    if not txt_path.exists():
        return sv.Detections.empty()

    image = model.load_image(image_path)
    detections = [
        [float(x) for x in line.split(" ") if x]
        for line in txt_path.read_text().splitlines()
        if line
    ]
    if len(detections) == 0 or len(detections[0]) == 0:
        return sv.Detections.empty()

    xywhs = [detection[1:] for detection in detections]
    xyxys = [xywh_to_xyxy(image, xywh) for xywh in xywhs]
    mask = [xyxy_to_mask(image, xyxy) for xyxy in xyxys]

    return sv.Detections(
        xyxy=np.array(xyxys),
        mask=np.array(mask).astype(bool),
        class_id=np.array([0] * len(xyxys)),
    )


def get_image_paths(folder: Path) -> list[Path]:
    return list(folder.glob("images/*"))
