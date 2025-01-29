from pathlib import Path

from PIL import Image
from supervision.detection.core import Detections

from action_labeler.v2.base import IFilter, IPreprocessor, IPrompt, IVisionLanguageModel

from .base import BaseActionLabeler


class MultipleImageActionLabeler(BaseActionLabeler):
    large_image_preprocessors: list[IPreprocessor]

    def __init__(
        self,
        folder: Path,
        prompt: IPrompt,
        model: IVisionLanguageModel,
        filters: list[IFilter],
        preprocessors: list[IPreprocessor],
        verbose: bool = False,
        save_every: int = 50,
        save_filename: str = "classification.pickle",
        large_image_preprocessors: list[IPreprocessor] = [],
    ):
        super().__init__(
            folder,
            prompt,
            model,
            filters,
            preprocessors,
            verbose,
            save_every,
            save_filename,
        )
        self.large_image_preprocessors = large_image_preprocessors

    def preprocess(
        self,
        image: Image.Image,
        index: int,
        detections: Detections,
        image_path: Path,
    ) -> list[Image.Image]:
        cropped_image = super().preprocess(image, index, detections, image_path)[0]
        full_sized_image = self.get_full_sized_image(image_path, index, detections)

        return [full_sized_image, cropped_image]

    def get_full_sized_image(
        self, image_path: Path, index: int, detections: Detections
    ) -> Image.Image:
        image = self.model.load_image(image_path)
        for preprocessor in self.large_image_preprocessors:
            image = preprocessor.preprocess(image, index, detections)
        return image
