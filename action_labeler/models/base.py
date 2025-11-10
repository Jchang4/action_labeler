from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
from transformers.image_utils import load_image


class IVisionLanguageModel(ABC):
    system_prompt: str

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt.strip().strip("\n")

    @abstractmethod
    def predict(self, prompt: str, images: list[Image.Image]) -> str: ...

    @abstractmethod
    def load_image(self, image_path: Path) -> Image.Image: ...


class BaseVisionLanguageModel(IVisionLanguageModel):
    def load_image(self, image_path: Path) -> Image.Image:
        return load_image(Image.open(image_path))
