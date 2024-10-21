import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import supervision as sv
from PIL import Image
from tqdm.auto import tqdm

from .helpers import (
    create_dataset_yaml,
    get_image,
    load_pickle,
    parse_response,
    save_pickle,
    xyxy_to_xywh,
)


class BasePrompt(ABC):
    actions: list[str]

    @abstractmethod
    def prompt(self) -> str: ...


class BaseClassificationModel(ABC):
    @abstractmethod
    def predict(self, images: list[Image.Image], prompt: str) -> str: ...


class BaseImageFilter(ABC):
    @abstractmethod
    def is_valid(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> bool: ...


class BaseImagePreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, image: Image.Image, index: int, detections: sv.Detections
    ) -> Image.Image: ...


class BaseActionLabeler(ABC):
    folder: Path
    prompt: BasePrompt
    model: BaseClassificationModel
    filters: list[BaseImageFilter]
    preprocessors: list[BaseImagePreprocessor]
    results: dict[str, dict[str, list[dict[str, str]]]]

    verbose: bool = False
    save_every: int = 50

    def __init__(
        self,
        folder: Path,
        prompt: BasePrompt,
        model: BaseClassificationModel,
        filters: list[BaseImageFilter],
        preprocessors: list[BaseImagePreprocessor],
        verbose: bool = False,
        save_every: int = 50,
    ):
        self.folder = folder
        self.prompt = prompt
        self.model = model
        self.filters = filters
        self.preprocessors = preprocessors
        self.results = load_pickle(folder, filename="classification.pickle")

        self.verbose = verbose
        self.save_every = save_every

    @abstractmethod
    def img_path_to_detections(
        self, image: Image.Image, img_path: Path
    ) -> sv.Detections:
        """Convert image path to detections."""
        raise NotImplementedError()

    def label(self):
        """Label images."""
        print(f"Starting with {len(self.results)} images")

        prompt = self.prompt.prompt()
        print(prompt)

        image_paths = self.images_path.iterdir()
        # image_paths = np.random.permutation(list(image_paths))
        image_paths = sorted(image_paths)

        for img_path in tqdm(image_paths, total=len(image_paths)):
            if not img_path.exists():
                continue

            # Raw Data - we reuse this later
            image = get_image(img_path)
            detections = self.img_path_to_detections(image, img_path)
            if detections.is_empty():
                continue

            for i in range(len(detections.xyxy)):
                xywh = xyxy_to_xywh(image, detections.xyxy[i])
                box_key = " ".join(map(str, xywh))

                # Skip if already labeled
                if (
                    str(img_path) in self.results
                    and box_key in self.results[str(img_path)]
                ):
                    continue

                # Skip detection
                is_valid = True
                for filter in self.filters:
                    if not filter.is_valid(image, i, detections):
                        is_valid = False
                        break
                if not is_valid:
                    continue

                # Preprocess Image
                annotated_frame = image.copy()
                for prepreprocessor in self.preprocessors:
                    annotated_frame = prepreprocessor.preprocess(
                        annotated_frame, i, detections
                    )

                # Predict
                try:
                    raw_response = self.model.predict(
                        [annotated_frame], self.prompt.prompt()
                    )
                    output = parse_response(raw_response)
                    # Save Results
                    self.results[str(img_path)][box_key] = output

                except json.JSONDecodeError:
                    print("Error parsing response")
                    print(raw_response)
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                if self.verbose:
                    annotated_frame.show()
                    print(output)

            if len(self.results) % self.save_every == 0:
                save_pickle(
                    dict(self.results),
                    self.folder,
                    filename="classification.pickle",
                )
                print(f"Saved {len(self.results)} images")

        self.remove_invalid_classes()
        save_pickle(
            dict(self.results),
            self.folder,
            filename="classification.pickle",
        )

    @property
    def images_path(self):
        """Get the path to the images folder."""
        return self.folder / "images"

    @property
    def segment_path(self):
        """Get the path to the segmentation folder."""
        return self.folder / "segment"

    @property
    def detect_path(self):
        """Get the path to the detection folder."""
        return self.folder / "detect"

    def get_segment_path(self, img_path: Path) -> Path:
        """Get the path to the segmentation file."""
        return self.segment_path / img_path.with_suffix(".txt").name

    def get_detect_path(self, img_path: Path) -> Path:
        """Get the path to the detection file."""
        return self.detect_path / img_path.with_suffix(".txt").name

    def create_dataset(self, output_folder: Optional[Path] = None):
        """Create a Yolo v8 dataset from the results."""
        if output_folder is None:
            folder_name = (
                str(self.folder).rsplit("datasets/", maxsplit=1)[-1].split("/")
            )
            folder_name = "_".join(folder_name)
            output_folder = Path("datasets") / (folder_name + "_autodistill")

        assert "datasets" in str(output_folder), "Output folder must be in datasets"

        # Create output folder
        shutil.rmtree(output_folder, ignore_errors=True)
        (output_folder / "train" / "images").mkdir(parents=True, exist_ok=True)
        (output_folder / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (output_folder / "valid" / "images").mkdir(parents=True, exist_ok=True)
        (output_folder / "valid" / "labels").mkdir(parents=True, exist_ok=True)

        # Make data.yaml
        classes = [c.lower() for c in self.prompt.actions]
        create_dataset_yaml(output_folder, classes)

        for img_path, box_key_to_label in self.results.items():
            img_path = Path(img_path)
            dataset = "train" if np.random.rand() < 0.8 else "valid"

            if not img_path.exists():
                print(f"Skipping non-existent image: {img_path}")
                continue

            for box_key, labels in box_key_to_label.items():
                action = labels[0]["Action"].lower()
                if action not in classes:
                    print(f"Skipping invalid class: {action}")
                    continue
                action_index = classes.index(action)

                # Copy image file
                shutil.copy(img_path, output_folder / dataset / "images")

                # Create txt file if doesn't exist
                txt_path = (
                    output_folder
                    / dataset
                    / "labels"
                    / img_path.with_suffix(".txt").name
                )
                if not txt_path.exists():
                    txt_path.touch()

                # Append to txt file
                with open(txt_path, "a") as f:
                    f.write(f"{action_index} {box_key}\n")

    def get_classes(self) -> set[str]:
        """Get classes from the results."""
        result_classes = set()
        for box_key_to_label in self.results.values():
            for labels in box_key_to_label.values():
                result_classes.update([label["Action"] for label in labels])
        return result_classes

    def get_class_counts(self) -> pd.DataFrame:
        """Get class counts from the results."""
        class_counts = {}
        for box_key_to_label in self.results.values():
            for labels in box_key_to_label.values():
                for label in labels:
                    action = label["Action"]
                    class_counts[action] = class_counts.get(action, 0) + 1

        return pd.DataFrame(class_counts.items(), columns=["Action", "Count"])

    ##########################
    #### Edit Results ########
    ##########################

    def remove_classes(self, classes: set[str]):
        """Remove classes from the results."""
        for img_path, box_key_to_label in self.results.items():
            for box_key, labels in box_key_to_label.items():
                # Remove classes
                box_key_to_label[box_key] = [
                    label for label in labels if label["Action"] not in classes
                ]

            # Remove empty box keys
            self.results[img_path] = {
                box_key: labels
                for box_key, labels in box_key_to_label.items()
                if labels
            }

    def remove_invalid_classes(self):
        """Remove invalid classes from the results."""
        valid_classes = self.prompt.actions
        all_classes = set()

        for _, box_key_to_label in self.results.items():
            for _, labels in box_key_to_label.items():
                all_classes.update([label["Action"] for label in labels])

        invalid_classes = all_classes - set(valid_classes)
        self.remove_classes(invalid_classes)

    ##########################
    #### Combine Results #####
    ##########################
    def combine_results(
        self,
        results: dict[str, dict[str, list[dict[str, str]]]],
        classes_to_keep: Optional[set[str]] = None,
    ):
        """Combine results."""
        for img_path, box_key_to_label in results.items():
            # Overwrite results with new results per image
            self.results[img_path] = {}
            for box_key, labels in box_key_to_label.items():
                # Filter classes
                if classes_to_keep is None:
                    self.results[img_path][box_key] = labels
                else:
                    self.results[img_path][box_key] = [
                        label
                        for label in labels
                        if label["Action"] in classes_to_keep
                        or label["Action"].lower() in classes_to_keep
                    ]
