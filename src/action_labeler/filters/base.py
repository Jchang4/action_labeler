from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from action_labeler.types import Detection


class BaseFilter(ABC):
    """Base interface for filters that exclude detections or images from processing."""

    @abstractmethod
    def filter(self, image: Image.Image, detections: list[Detection]) -> bool:
        """Decide whether an image should be processed.

        Args:
            image: The input image.
            detections: The detections for this image.

        Returns:
            True if the image should be kept, False if it should be excluded.
        """
        ...
