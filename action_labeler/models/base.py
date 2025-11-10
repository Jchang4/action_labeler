from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
from transformers.image_utils import load_image

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that can identify the action of each person in an image. You are thorough and detailed. You only choose actions from the list below. You only choose actions where you can see the person doing the action and the object they are interacting with. You do not choose actions where the person is not doing anything or the object is not visible"


class IVisionLanguageModel(ABC):
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @abstractmethod
    def predict(self, prompt: str, images: list[Image.Image]) -> str: ...

    @abstractmethod
    def load_image(self, image_path: Path) -> Image.Image: ...


class BaseVisionLanguageModel(IVisionLanguageModel):
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt
        super().__init__()

    def load_image(self, image_path: Path) -> Image.Image:
        return load_image(Image.open(image_path))
