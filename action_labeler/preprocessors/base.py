from abc import ABC, abstractmethod

from PIL import Image

from action_labeler.detections.detection import Detection


class IPreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, image: Image.Image, index: int, detections: Detection
    ) -> Image.Image: ...
