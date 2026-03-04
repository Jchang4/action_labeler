from abc import ABC, abstractmethod

from PIL import Image


class BasePreprocessor(ABC):
    """Base interface for image preprocessors applied before VLM inference."""

    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        """Transform an image before it is sent to the model.

        Args:
            image: The input image.

        Returns:
            The transformed image.
        """
        ...
