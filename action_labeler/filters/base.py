from abc import ABC, abstractmethod

from PIL import Image

from action_labeler.detections.detection import Detection


class IFilter(ABC):
    @abstractmethod
    def is_valid(
        self, image: Image.Image, index: int, detections: Detection
    ) -> bool: ...
