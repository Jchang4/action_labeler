from abc import ABC, abstractmethod

from PIL import Image


class BaseModel(ABC):
    """Base interface for vision-language models used by ActionLabeler."""

    @abstractmethod
    def load_image(self, image: Image.Image) -> Image.Image:
        """Preprocess an image for this model.

        Many models require specific image formats (size, color mode, etc.).
        Override this to apply model-specific preprocessing.

        The default implementation returns the image unchanged.
        """
        ...

    @abstractmethod
    def predict(self, system: str, user: str, images: list[Image.Image]) -> str:
        """Run inference on the given prompt and images.

        Args:
            system: The system prompt.
            user: The user message.
            images: Pre-processed images (already passed through load_image).

        Returns:
            The model's text response.
        """
        ...
