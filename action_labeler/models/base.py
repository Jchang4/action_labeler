from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
from transformers.image_utils import load_image


class IVisionLanguageModel(ABC):
    @abstractmethod
    def predict(self, prompt: str, images: list[Image.Image]) -> str: ...

    @abstractmethod
    def load_image(self, image_path: Path) -> Image.Image: ...


class BaseVisionLanguageModel(IVisionLanguageModel):
    def load_image(self, image_path: Path) -> Image.Image:
        return load_image(Image.open(image_path))
