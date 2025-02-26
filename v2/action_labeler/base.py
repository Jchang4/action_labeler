from pathlib import Path

import supervision as sv
from matplotlib import pyplot as plt
from PIL import Image
from supervision.detection.core import Detections
from tqdm.auto import tqdm

from action_labeler.helpers import (
    get_image_paths,
    load_pickle,
    save_pickle,
    xyxy_to_xywh,
)
from action_labeler.v2.base import (
    IActionLabeler,
    IFilter,
    IPreprocessor,
    IPrompt,
    IVisionLanguageModel,
)

from .action_labeler_dataset import ActionLabelDataset
from .helpers import image_to_detect_path, image_to_detections


class BaseActionLabeler(IActionLabeler):
    folder: Path
    prompt: IPrompt
    model: IVisionLanguageModel
    filters: list[IFilter]
    preprocessors: list[IPreprocessor]
    results: dict[str, dict[str, list[dict[str, str]]]]
    labels: ActionLabelDataset

    # Options
    verbose: bool = False
    save_every: int = 50
    save_filename: str = "classification.pickle"

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
    ):
        self.folder = folder
        self.prompt = prompt
        self.model = model
        self.filters = filters
        self.preprocessors = preprocessors
        self.results = load_pickle(folder, filename=save_filename)
        self.labels = ActionLabelDataset(dataset_path=folder, file_name=save_filename)

        # Options
        self.verbose = verbose
        self.save_every = save_every
        self.save_filename = save_filename

    def label(self):
        print(f"Starting with {len(self.results)} labeled images")

        image_paths = get_image_paths(self.folder)
        for image_path in tqdm(image_paths):
            detect_path = image_to_detect_path(image_path)
            if not detect_path.exists():
                print(f"Missing detection for {str(image_path)}")
                continue

            detections = self.img_path_to_detections(image_path, self.model)
            if detections.is_empty():
                print(f"No detections for {str(image_path)}")
                continue

            for i, box in enumerate(detections.xyxy):
                image = self.model.load_image(image_path)
                box_key = " ".join(map(str, xyxy_to_xywh(image, box)))

                if (
                    str(image_path) in self.results
                    and box_key in self.results[str(image_path)]
                ):
                    continue

                # Ensure image is valid
                if not self.is_valid(image, i, detections):
                    if box_key in self.results[str(image_path)]:
                        del self.results[str(image_path)][box_key]
                    continue

                # Preprocess image
                images = self.preprocess(image, i, detections, image_path)

                # Predict each detected object
                prediction = self.predict(images, i, detections, image_path)
                if prediction is None:
                    continue

                # Save results
                self.results[str(image_path)][box_key] = prediction

                if self.verbose:
                    self.show_prediction(images, prediction)

            if len(self.results) % self.save_every == 0:
                self.save_results()
                print(f"Saved {len(self.results)} images")

        self.save_results()
        print(f"Saved {len(self.results)} images")

    def img_path_to_detections(
        self, img_path: Path, model: IVisionLanguageModel
    ) -> Detections:
        return image_to_detections(img_path, model)

    def is_valid(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> bool:
        """Check if the detection is valid using filters"""
        for filter in self.filters:
            if not filter.is_valid(image, index, detections):
                return False
        return True

    def preprocess(
        self,
        image: Image.Image,
        index: int,
        detections: sv.Detections,
        image_path: Path,
    ) -> list[Image.Image]:
        """Augment the image using preprocessors"""
        image = image.copy()
        for prepreprocessor in self.preprocessors:
            image = prepreprocessor.preprocess(image, index, detections)
        return [image]

    def predict(
        self,
        images: list[Image.Image],
        index: int,
        detections: sv.Detections,
        image_path: Path,
    ) -> str | None:
        try:
            prompt = self.prompt.prompt(images[0], index, detections, image_path)
            if self.verbose:
                print(prompt)
            prediction = self.model.predict(prompt, images)
            return prediction
        except Exception as e:
            print(f"Error: {e}")
        return None

    def save_results(self):
        save_pickle(
            dict(self.results),
            self.folder,
            filename=self.save_filename,
        )

        # self.labels.save()

    def show_prediction(self, images: list[Image.Image], prediction: str):
        # Display images in a horizontal row
        fig, ax = plt.subplots(1, len(images), figsize=(15, 5))
        for i, image in enumerate(images):
            (ax if len(images) == 1 else ax[i]).imshow(image)
        plt.show()
        print(prediction)
