from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from .dataset import Dataset
from .filters.base import BaseFilter
from .models.base import BaseModel as BaseVLM
from .preprocessors.base import BasePreprocessor
from .prompts import Prompt
from .types import Detection

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class LabelResult:
    """Result of labeling a single detection."""

    image_path: Path
    detection: Detection
    response: BaseModel | str


class ActionLabeler(ABC):
    """Abstract base for labeling pipelines.

    Subclasses implement label() to define how detections map to model calls.
    The shared run() method handles file loading, filtering, and error handling.
    """

    def __init__(
        self,
        model: BaseVLM,
        prompt: Prompt,
        preprocessors: list[BasePreprocessor] | None = None,
        filters: list[BaseFilter] | None = None,
    ):
        self.model = model
        self.prompt = prompt
        self.preprocessors = preprocessors or []
        self.filters = filters or []

    def run(self, dataset_path: Path) -> Dataset:
        """Orchestrate the labeling pipeline over a dataset directory.

        For each image:
        1. Load image + detection file
        2. Apply filters — skip image if any filter rejects
        3. Delegate to label() (subclass-defined strategy)
        4. Collect results
        5. On error: print image_path + exception, continue

        Returns a Dataset backed by a pandas DataFrame.
        """
        results: list[LabelResult] = []
        image_paths = self._load_images(dataset_path)

        for image_path in image_paths:
            try:
                image = Image.open(image_path)
                image = self.model.load_image(image)

                detection_path = (
                    dataset_path / "detect" / f"{image_path.stem}.txt"
                )
                detections = self._load_detections(detection_path, image)

                if not self._apply_filters(image, detections):
                    continue

                label_results = self.label(image, detections)
                for result in label_results:
                    result.image_path = image_path
                results.extend(label_results)
            except Exception as e:
                print(f"{image_path}: {e}")

        return Dataset.from_label_results(results)

    @abstractmethod
    def label(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[LabelResult]:
        """Subclasses define how detections map to model calls.

        Receives the raw image (not yet preprocessed) and all detections
        for that image. Responsible for calling preprocessors, model.predict(),
        and prompt.parse() as needed.

        Returns one LabelResult per detection that was labeled.
        """
        ...

    def _load_images(self, dataset_path: Path) -> list[Path]:
        """Glob for image files in dataset_path/images/."""
        images_dir = dataset_path / "images"
        paths = [
            p
            for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return paths

    def _load_detections(
        self, detection_path: Path, image: Image.Image
    ) -> list[Detection]:
        """Parse a YOLO-format detection txt file into Detection objects."""
        return Detection.load_txt(detection_path, image)

    def _apply_filters(
        self, image: Image.Image, detections: list[Detection]
    ) -> bool:
        """Return True if image passes all filters."""
        return all(f.filter(image, detections) for f in self.filters)

    def _apply_preprocessors(
        self, image: Image.Image, detections: list[Detection]
    ) -> Image.Image:
        """Apply preprocessors in order, return transformed image."""
        for preprocessor in self.preprocessors:
            image = preprocessor.process(image, detections)
        return image
