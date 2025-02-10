import supervision as sv
from PIL.Image import Image

from action_labeler.helpers import get_detection
from action_labeler.v2.base import IPreprocessor


class MaskPreprocessor(IPreprocessor):
    """Apply mask for current detection."""

    annotator: sv.MaskAnnotator

    def __init__(self, opacity: float = 0.5, color: sv.Color = sv.Color.BLACK):
        self.annotator = sv.MaskAnnotator(opacity=opacity, color=color)

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: sv.Detections,
    ) -> Image:
        single_detection = get_detection(
            xyxy=[detections.xyxy[index]],
            mask=[detections.mask[index]],
        )
        return self.annotator.annotate(
            scene=image,
            detections=single_detection,
        )


class BackgroundMaskPreprocessor(IPreprocessor):
    """Apply mask for current detection."""

    annotator: sv.MaskAnnotator

    def __init__(self, opacity: float = 0.5, color: sv.Color = sv.Color.BLACK):
        self.annotator = sv.MaskAnnotator(opacity=opacity, color=color)

    def preprocess(
        self,
        image: Image,
        index: int,
        detections: sv.Detections,
    ) -> Image:
        single_detection = get_detection(
            xyxy=[detections.xyxy[index]],
            mask=[~detections.mask[index].astype(bool)],
        )
        return self.annotator.annotate(
            scene=image,
            detections=single_detection,
        )
