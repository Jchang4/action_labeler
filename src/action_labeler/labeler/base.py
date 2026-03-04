from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from ..dataset import Dataset
from ..filters.base import BaseFilter
from ..models.base import BaseModel as BaseVLM
from ..preprocessors.base import BasePreprocessor
from ..prompts import Prompt
from ..types import Detection

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ActionLabeler(ABC):
    """Abstract base for labeling pipelines.

    Subclasses implement label() to define how detections map to model calls.
    The shared run() method handles file loading, filtering, and error handling.

    Args:
        model: Vision-language model for inference.
        prompt: Prompt template for system/user messages and response parsing.
        preprocessors: Image preprocessing chains. Each inner list is a chain
            of preprocessors that produces one image. Multiple chains produce
            multiple images sent to the model together.
            Example: ``[[crop, resize]]`` produces 1 image.
            Example: ``[[crop], [mask], [bbox]]`` produces 3 images.
        filters: Filters that can reject entire images from processing.
    """

    def __init__(
        self,
        model: BaseVLM,
        prompt: Prompt,
        preprocessors: list[list[BasePreprocessor]] | None = None,
        filters: list[BaseFilter] | None = None,
    ):
        self.model = model
        self.prompt = prompt
        self.preprocessors = preprocessors or []
        self.filters = filters or []

    def run(
        self, dataset_path: Path, dataset: Dataset | None = None
    ) -> Dataset:
        """Orchestrate the labeling pipeline over a dataset directory.

        For each image:
        1. Load image + detection file
        2. Apply filters — skip image if any filter rejects
        3. Skip detections already in dataset (enables resume)
        4. Delegate to label() (subclass-defined strategy)
        5. Add results to dataset incrementally
        6. On error: print image_path + exception, continue

        Args:
            dataset_path: Path to the dataset directory.
            dataset: Optional existing Dataset to resume from. Already-labeled
                detections are skipped.

        Returns a Dataset backed by a pandas DataFrame.
        """
        if dataset is None:
            dataset = Dataset()
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

                # Skip fully-labeled images
                if all(
                    dataset.has_row(image_path, d) for d in detections
                ):
                    continue

                responses = self.label(image, detections)
                dataset.add_rows(image_path, detections, responses)
            except Exception as e:
                print(f"{image_path}: {e}")

        return dataset

    @abstractmethod
    def label(
        self, image: Image.Image, detections: list[Detection]
    ) -> list[BaseModel | str]:
        """Subclasses define how detections map to model calls.

        Receives the raw image (not yet preprocessed) and all detections
        for that image. Responsible for calling preprocessors, model.predict(),
        and prompt.parse() as needed.

        Returns one response per detection, positionally matched.
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
    ) -> list[Image.Image]:
        """Apply each preprocessor chain to produce one image per chain.

        Returns a list of images, one per preprocessing chain. If no
        preprocessors are configured, returns the original image in a list.
        """
        if not self.preprocessors:
            return [image]

        images = []
        for chain in self.preprocessors:
            img = image.copy()
            for preprocessor in chain:
                img = preprocessor.process(img, detections)
            images.append(img)
        return images
