from pathlib import Path

from PIL import Image
from transformers.image_utils import load_image

from action_labeler.v2.base import IVisionLanguageModel


class BaseVisionLanguageModel(IVisionLanguageModel):
    def load_image(self, image_path: Path) -> Image.Image:
        return load_image(Image.open(image_path))
