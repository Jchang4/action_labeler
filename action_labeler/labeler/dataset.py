from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from action_labeler.dataclasses import DetectionType
from action_labeler.detections import Detection
from action_labeler.helpers.detections_helpers import xywh_to_xyxy
from action_labeler.helpers.image_helpers import add_bounding_boxes
from action_labeler.helpers.yolov8_dataset import (
    add_group_to_dataset,
    create_dataset_folder,
)


class LabelerDataset:
    folder: Path
    filename: str
    classes: list[str]

    # Class variables
    df: pd.DataFrame

    def __init__(
        self,
        folder: Path,
        classes: list[str],
        filename: str = "classification.pickle",
    ):
        self.folder = Path(folder)
        self.filename = filename
        self.classes = classes
        self._load_df()

    def _load_df(self):
        if (self.folder / self.filename).exists():
            self.df = pd.read_pickle(self.folder / self.filename)
        else:
            self.df = pd.DataFrame(
                columns=["image_path", "xywh", "segmentation_points", "action"]
            )

    def add_row(
        self,
        image_path: Path,
        xywh: list[float],
        segmentation_points: list[list[float]],
        action: str,
    ) -> "LabelerDataset":
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    [
                        {
                            "image_path": str(image_path),
                            "xywh": xywh,
                            "segmentation_points": segmentation_points,
                            "action": action,
                        },
                    ]
                ),
            ]
        )
        return self

    def does_row_exist(self, image_path: Path | str, xywh: list[float]) -> bool:
        if str(image_path) not in self.df["image_path"].values:
            return False

        # Filter by image path first
        matching_rows = self.df[self.df["image_path"] == str(image_path)]

        # Then check if any row has the matching xywh
        xywh_str = " ".join(map(str, xywh))
        matching_xywh = (
            matching_rows["xywh"].apply(lambda x: " ".join(map(str, x))) == xywh_str
        )

        return matching_xywh.any()

    def save(self) -> "LabelerDataset":
        self.df.to_pickle(self.folder / self.filename)
        return self

    def merge_datasets(
        self, other_datasets: list["LabelerDataset"]
    ) -> "LabelerDataset":
        for other_dataset in other_datasets:
            self.df = pd.concat(
                [self.df, other_dataset.df],
                ignore_index=True,
                sort=False,
            )
        return self

    def create_yolo_v8_dataset(
        self,
        output_folder: str | Path,
        delete_existing: bool = False,
        detection_type: DetectionType = DetectionType.DETECT,
    ) -> "LabelerDataset":
        # Update self.classes with all possible class names in the dataframe
        # We must preserve the existing order of classes, and append new ones to the end
        for class_name in self.df["action"].unique():
            if class_name not in self.classes:
                self.classes.append(class_name)

        class_name_to_id = {class_name: i for i, class_name in enumerate(self.classes)}
        # Set the class_id in the dataframe
        self.df["class_id"] = self.df["action"].apply(lambda x: class_name_to_id[x])
        create_dataset_folder(output_folder, class_name_to_id, delete_existing)
        for image_path, group in tqdm(self.df.groupby("image_path")):
            add_group_to_dataset(
                image_path,
                group,
                dataset_folder=output_folder,
                detection_type=detection_type,
            )
        return self

    def plot_class_distribution(self) -> "LabelerDataset":
        class_counts = self.df["action"].value_counts()
        class_counts.plot(kind="bar")
        plt.show()

        return self

    def plot_image(self, image_path: Path) -> "LabelerDataset":
        image = Image.open(image_path)
        plt.imshow(image)
        plt.show()

        return self

    def plot_images_for_class(
        self, class_name: str, num_images: int = 5
    ) -> "LabelerDataset":
        class_df = self.df[self.df["action"] == class_name]
        # Create a grid of images (3 columns per row)
        num_rows = int(np.ceil(min(num_images, len(class_df)) / 3))
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))

        # Handle case where there's only one row
        if num_rows == 1:
            axes = [axes]

        # Sample images for the class
        sampled_paths = (
            class_df["image_path"]
            .sample(
                min(num_images, len(class_df["image_path"].unique())),
                replace=False,
            )
            .tolist()
        )

        # Plot each image in the grid
        for i, image_path in enumerate(sampled_paths):
            row, col = i // 3, i % 3
            image = Image.open(image_path)

            image_df = class_df[class_df["image_path"] == image_path]
            group = image_df["xywh"].tolist()
            class_id = (
                image_df["action"]
                .apply(lambda x: self.classes.index(x) if x in self.classes else 0)
                .iloc[0]
            )
            xyxys = [xywh_to_xyxy(xywh, image.size) for xywh in group]
            detections = Detection(
                xyxy=xyxys, mask=None, class_id=class_id, image_size=image.size
            )
            image_with_boxes = add_bounding_boxes(image, detections)

            axes[row][col].imshow(image_with_boxes)
            axes[row][col].set_title(f"{Path(image_path).name}")
            axes[row][col].axis("off")

        # Hide unused subplots
        for i in range(len(sampled_paths), num_rows * 3):
            row, col = i // 3, i % 3
            axes[row][col].axis("off")

        plt.tight_layout()
        plt.show()

        return self

    def __repr__(self):
        return f"LabelerDataset(folder={self.folder}, filename={self.filename}, classes={len(self.classes)})"

    def __len__(self):
        return len(self.df)
