from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from action_labeler.types import Detection


class BasePreprocessor(ABC):
    """Base interface for image preprocessors applied before VLM inference."""

    @abstractmethod
    def process(self, image: Image.Image, detections: list[Detection]) -> Image.Image:
        """Transform an image before it is sent to the model.

        Args:
            image: The input image.
            detections: The detections for this image.

        Returns:
            The transformed image.
        """
        ...
