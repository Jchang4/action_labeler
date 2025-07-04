import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter
from action_labeler.helpers import get_image_paths, image_to_txt_path, load_image
from action_labeler.models.base import IVisionLanguageModel
from action_labeler.preprocessors.base import IPreprocessor
from action_labeler.prompt.base import BasePrompt


class ActionLabeler:
    folder: Path
    prompt: BasePrompt
    model: IVisionLanguageModel
    filters: list[IFilter]
    preprocessors: list[IPreprocessor]

    # Options
    verbose: bool = False
    save_every: int = 50
    save_filename: str = "classification.pickle"

    # Class variables
    results: pd.DataFrame

    def __init__(
        self,
        folder: Path,
        prompt: BasePrompt,
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

        self.verbose = verbose
        self.save_every = save_every
        self.save_filename = save_filename

        # Load results file if it exists otherwise create an empty dataframe
        self.results = self.load_results()

    def label(self):
        print(f"Starting with {len(self.results)} labeled images")

        image_paths = get_image_paths(self.folder)

        for image_path in tqdm(image_paths):
            image = load_image(image_path)
            txt_path = image_to_txt_path(image_path)
            if not txt_path.exists():
                print(f"No detections for {str(image_path)}")
                continue

            detections = Detection.from_text_path(txt_path, image.size)
            if detections.is_empty():
                print(f"No detections for {str(image_path)}")
                continue

            for i in range(len(detections.xyxy)):
                if not self.apply_filters(image, i, detections):
                    continue
                elif self.does_result_exist(image_path, i, detections):
                    continue

                preprocessed_image = self.apply_preprocessors(image, i, detections)

                raw_model_output = self.apply_model(
                    image_path, [preprocessed_image], i, detections
                )
                self.add_results(
                    image_path,
                    detections.xywhn[i],
                    raw_model_output,
                )

                if (
                    self.verbose
                    and len(self.results) % self.save_every == 0
                    and i == len(detections.xyxy) - 1
                ):
                    # Show the last detection
                    image.show()
                    preprocessed_image.show()
                    print(raw_model_output)

            if len(self.results) % self.save_every == 0:
                self.save_results()
                print(f"Saved {len(self.results)} images")

        self.save_results()
        print(f"Saved {len(self.results)} images")

    def apply_filters(self, image: Image.Image, index: int, detections: Detection):
        for filter in self.filters:
            if not filter.is_valid(image, index, detections):
                return False
        return True

    def apply_preprocessors(
        self, image: Image.Image, index: int, detections: Detection
    ):
        for preprocessor in self.preprocessors:
            image = preprocessor.preprocess(image, index, detections)
        return image

    def apply_model(
        self,
        image_path: Path | str,
        images: list[Image.Image],
        index: int,
        detections: Detection,
    ):
        image_path = Path(image_path)
        prompt = self.prompt.prompt(index, detections, image_path)
        return self.model.predict(prompt, images)

    def save_results(self):
        self.results.to_pickle(self.folder / self.save_filename)

    def load_results(self):
        try:
            return pd.read_pickle(self.folder / self.save_filename)
        except FileNotFoundError:
            return pd.DataFrame(columns=["image_path", "xywh", "model_output"])

    def does_result_exist(
        self, image_path: Path | str, index: int, detections: Detection
    ) -> bool:
        image_path = Path(image_path)
        return (
            not self.results.empty
            and len(
                self.results[(self.results["image_path"] == image_path)][
                    self.results["xywh"].apply(lambda x: " ".join(map(str, x)))
                    == " ".join(map(str, detections.xywhn[index]))
                ]
            )
            > 0
        )

    def add_results(self, image_path: Path | str, xywh: list[float], model_output: str):
        image_path = Path(image_path)
        self.results = pd.concat(
            [
                self.results,
                pd.DataFrame(
                    [
                        {
                            "image_path": image_path,
                            "xywh": xywh,
                            "model_output": model_output,
                        }
                    ]
                ),
            ]
        )

    def __repr__(self):
        return f"ActionLabeler(folder={self.folder})"
