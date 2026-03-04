from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from action_labeler.preprocessors.base import BasePreprocessor

if TYPE_CHECKING:
    from action_labeler.types import Detection


class Resize(BasePreprocessor):
    """Resize an image while preserving aspect ratio.

    The larger dimension is scaled to ``size``; the smaller dimension is
    scaled proportionally.  Images already at or below ``size`` are
    returned unchanged.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def process(self, image: Image.Image, detections: list[Detection]) -> Image.Image:
        width, height = image.size
        if max(width, height) <= self.size:
            return image

        scale = self.size / max(width, height)
        new_width = round(width * scale)
        new_height = round(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
