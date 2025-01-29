from abc import ABC, abstractmethod
from pathlib import Path

import supervision as sv
from PIL import Image


class IPrompt(ABC):
    @abstractmethod
    def prompt(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
        image_path: Path,
    ) -> str: ...


class IFilter(ABC):
    @abstractmethod
    def is_valid(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> bool: ...


class IPreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> Image.Image: ...


class IVisionLanguageModel(ABC):
    @abstractmethod
    def predict(self, prompt: str, images: list[Image.Image]) -> str: ...

    @abstractmethod
    def load_image(self, image_path: Path) -> Image.Image: ...


class IActionLabeler(ABC):
    folder: Path
    prompt: IPrompt
    model: IVisionLanguageModel
    filters: list[IFilter]
    preprocessors: list[IPreprocessor]
    results: dict[str, dict[str, list[dict[str, str]]]]

    # Options
    verbose: bool
    save_every: int
    save_filename: str

    @abstractmethod
    def label(self) -> str:
        """Label the images in the folder"""
        ...

    @abstractmethod
    def img_path_to_detections(
        self, img_path: Path, model: IVisionLanguageModel
    ) -> sv.Detections:
        """Convert image path to detections."""
        ...

    @abstractmethod
    def is_valid(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> bool:
        """Check if the detection is valid using filters"""
        ...

    @abstractmethod
    def preprocess(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
        image_path: Path,
    ) -> list[Image.Image]:
        """Augment the image using preprocessors"""
        ...

    @abstractmethod
    def predict(
        self,
        images: list[Image.Image],
        index: int,
        detections: sv.Detections,
        image_path: Path,
    ) -> str | None:
        """Predict the action for the detection"""
        ...

    @abstractmethod
    def save_results(self):
        """Save the results"""
        ...
