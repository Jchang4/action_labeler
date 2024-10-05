from pathlib import Path
import supervision as sv
from PIL import Image

from .base import BaseActionLabeler
from .helpers import (
    segmentation_to_xyxy,
    segmentation_to_mask,
    get_detection,
    xywh_to_xyxy,
    xyxy_to_mask,
)


class DetectActionLabeler(BaseActionLabeler):
    def img_path_to_detections(
        self, image: Image.Image, img_path: Path
    ) -> sv.Detections:
        txt_path = self.get_detect_path(img_path)
        if not txt_path.exists():
            return sv.Detections.empty()

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

        # Convert to Detections
        detections = get_detection(xyxys, mask)

        return detections


class SegmentationActionLabeler(BaseActionLabeler):
    def img_path_to_detections(
        self, image: Image.Image, img_path: Path
    ) -> sv.Detections:
        txt_path = self.get_segment_path(img_path)
        segmentations = [
            [float(x) for x in line.split(" ") if x]
            for line in txt_path.read_text().splitlines()
            if line
        ]
        if len(segmentations) == 0 or len(segmentations[0]) == 0:
            return sv.Detections.empty()

        xyxys = [segmentation_to_xyxy(image, segment) for segment in segmentations]
        mask = [segmentation_to_mask(image, segment) for segment in segmentations]

        # Convert to Detections
        detections = get_detection(xyxys, mask)

        return detections
