from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from action_labeler.dataclasses import DetectionType
from action_labeler.helpers.general import get_image_paths
from action_labeler.helpers.yolov8_dataset import (
    add_group_to_dataset_yolo_v8,
    create_dataset_folder,
    get_data_yaml,
    get_label_path,
    yolov8_labels_to_row,
)


class YoloV8Dataset:
    folder: Path
    df: pd.DataFrame
    classes: list[str]
    detection_type: DetectionType

    def __init__(
        self,
        folder: str | Path,
        classes: list[str],
        df: pd.DataFrame,
    ):
        self.folder = Path(folder)
        self.classes = classes
        self.class_name_to_id = {class_name: i for i, class_name in enumerate(classes)}
        self.df = df

    @classmethod
    def from_folder(cls, folder: str | Path) -> "YoloV8Dataset":
        folder = Path(folder)
        data_yaml = get_data_yaml(folder)
        classes = data_yaml["names"]

        # Load images and text files
        data = []
        for dataset in ["train", "valid"]:
            for image_path in get_image_paths(folder / dataset / "images"):
                label_path = get_label_path(image_path)
                if label_path is None:
                    continue

                label_rows = yolov8_labels_to_row(label_path)
                for row in label_rows:
                    data.append(
                        {
                            "dataset": dataset,
                            "image_path": image_path,
                            "xywh": row[1:],
                            "class_id": int(row[0]),
                        }
                    )

        df = pd.DataFrame(data)
        return cls(folder, classes, df)

    @classmethod
    def empty(cls, folder: str | Path, classes: list[str]) -> "YoloV8Dataset":
        folder = Path(folder)
        return cls(
            folder,
            classes,
            pd.DataFrame(columns=["dataset", "image_path", "xywh", "class_id"]),
        )

    def save(self, output_folder: str | Path, delete_existing: bool = False):
        self.df["image_name"] = self.df["image_path"].apply(lambda x: Path(x).name)
        create_dataset_folder(output_folder, self.class_name_to_id, delete_existing)
        for image_path, group in tqdm(self.df.groupby("image_path")):
            add_group_to_dataset_yolo_v8(
                image_path,
                group,
                dataset_folder=output_folder,
            )

    def remap_classes(self, old_to_new_class_name: dict[str, str]):
        old_to_new_class_id = {}
        for old_class_name, new_class_name in tqdm(old_to_new_class_name.items()):
            new_class_id = self.class_name_to_id.get(new_class_name)
            if new_class_id is None:
                self.classes.append(new_class_name)
                self.class_name_to_id[new_class_name] = len(self.classes) - 1
                new_class_id = self.class_name_to_id[new_class_name]
            old_to_new_class_id[self.class_name_to_id[old_class_name]] = new_class_id

        self.df["class_id"] = self.df["class_id"].map(old_to_new_class_id)
        self.df["class_id"] = self.df["class_id"].astype(int)

    def delete_classes(self, class_names: list[str]):
        original_classes = self.classes.copy()

        # First, remove the classes and their data
        for class_name in tqdm(class_names):
            if class_name not in self.class_name_to_id:
                print(f"'{class_name}' is not in the dataset")
                continue
            class_id = self.class_name_to_id[class_name]
            self.df = self.df[self.df["class_id"] != class_id]
            del self.class_name_to_id[class_name]
            self.classes.remove(class_name)

        # Now rebuild the class_name_to_id mapping to ensure continuous IDs
        self.class_name_to_id = {name: idx for idx, name in enumerate(self.classes)}

        # Update all class_ids in the dataframe to match the new mapping
        class_id_mapping = {}
        for i, row in self.df.iterrows():
            old_class_id = row["class_id"]
            class_name = original_classes[old_class_id]
            new_class_id = self.class_name_to_id[class_name]
            class_id_mapping[old_class_id] = new_class_id

        # Apply the mapping to update all class_ids
        self.df["class_id"] = self.df["class_id"].map(class_id_mapping)
        self.df["class_id"] = self.df["class_id"].astype(int)

    def create_balanced_dataset(
        self, min_samples: int | None = None, random_state: int | None = 42
    ):
        """
        Create a balanced dataset by ensuring all classes have the same number of samples.

        Args:
            min_samples (int): The minimum number of samples for each class. If min_samples is provided, the class distribution may not be even.
        """
        class_counts = self.df["class_id"].value_counts()
        min_samples = min_samples if min_samples is not None else min(class_counts)

        # Create a new dataframe with the balanced dataset
        data = []
        for class_id, class_name in tqdm(enumerate(self.classes)):
            class_df = self.df[self.df["class_id"] == class_id]
            class_df = (
                class_df.sample(min_samples, random_state=random_state).reset_index(
                    drop=True
                )
                if len(class_df) > min_samples
                else class_df
            )
            data.extend(class_df.to_dict(orient="records"))

        # Set dataset column 80% train, 20% valid
        balanced_ds = YoloV8Dataset(self.folder, self.classes, pd.DataFrame(data))
        # Extract image_name from image_path and create a new column
        balanced_ds.df["image_name"] = balanced_ds.df["image_path"].apply(
            lambda x: Path(x).name
        )

        # Get unique image names
        unique_images = balanced_ds.df["image_name"].unique()

        # Create a mapping of image_name to dataset (train or valid)
        image_to_dataset = {
            img: np.random.choice(["train", "valid"], p=[0.8, 0.2])
            for img in unique_images
        }

        # Assign dataset based on image_name
        balanced_ds.df["dataset"] = (
            balanced_ds.df["image_name"].map(image_to_dataset).astype(str)
        )
        return balanced_ds

    def add_background_images(
        self, background_images_folder: str | Path, pct_background: float = 0.2
    ):
        background_images_folder = Path(background_images_folder)
        background_images = get_image_paths(background_images_folder)
        np.random.shuffle(background_images)
        num_samples = int(self.df["class_id"].value_counts().min() * pct_background)
        for image_path in background_images[:num_samples]:
            dataset = np.random.choice(["train", "valid"], p=[0.8, 0.2])
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        {
                            "dataset": [dataset],
                            "image_path": [image_path],
                            "xywh": [None],
                            "class_id": [None],
                        }
                    ),
                ],
                ignore_index=True,
            )
        return self

    def plot_class_distribution(self):
        # Create a single figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get class names for all samples
        class_names = self.df["class_id"].apply(
            lambda x: self.classes[x] if x is not None else "background"
        )

        # Get counts for train and validation sets
        train_counts = class_names[self.df["dataset"] == "train"].value_counts()
        valid_counts = class_names[self.df["dataset"] == "valid"].value_counts()

        # Combine all class names to ensure we have all categories
        all_classes = pd.concat([train_counts, valid_counts]).index.unique()

        # Create a DataFrame with all classes and fill missing values with 0
        df_plot = pd.DataFrame(
            {
                "Train": [train_counts.get(cls, 0) for cls in all_classes],
                "Valid": [valid_counts.get(cls, 0) for cls in all_classes],
            },
            index=all_classes,
        )

        # Plot both distributions in one graph with different colors
        df_plot.plot(kind="bar", ax=ax, color=["blue", "orange"])

        ax.set_title("Class Distribution in Training and Validation Sets")
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")
        ax.legend(["Training Set", "Validation Set"])

        plt.tight_layout()
        plt.show()

    def plot_dataset(self):
        pass

    def plot_class(self):
        pass

    def __len__(self):
        return len(self.df)
