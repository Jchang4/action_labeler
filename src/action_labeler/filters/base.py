from abc import ABC, abstractmethod

from PIL import Image


class BaseFilter(ABC):
    """Base interface for filters that exclude detections or images from processing."""

    @abstractmethod
    def filter(self, image: Image.Image) -> bool:
        """Decide whether an image should be processed.

        Args:
            image: The input image.

        Returns:
            True if the image should be kept, False if it should be excluded.
        """
        ...
