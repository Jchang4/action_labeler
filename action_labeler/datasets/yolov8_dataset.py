import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from action_labeler.helpers.general import get_image_paths
from action_labeler.helpers.yolov8_dataset import (
    get_data_yaml,
    get_label_path,
    yolov8_labels_to_row,
)
from action_labeler.labeler.detection_labelers.dataset import LabelerDataset


def create_dataset_folder(
    folder: str | Path, class_name_to_id: dict, delete_existing: bool = False
) -> Path:
    """Create a dataset folder for a given folder.

    Dataset folder structure:
    - folder_name/
        - train/
            - images/
            - labels/
        - valid/
            - images/
            - labels/
        - data.yaml

    Args:
        folder (str | Path): The folder to create the dataset folder for.
        delete_existing (bool): Whether to delete the existing dataset folder.

    Returns:
        Path: The path to the dataset folder.
    """

    folder = Path(folder)

    if delete_existing:
        shutil.rmtree(folder, ignore_errors=True)

    folder.mkdir(parents=True, exist_ok=True)

    for dataset in ["train", "valid"]:
        for subfolder in ["images", "labels"]:
            (folder / dataset / subfolder).mkdir(parents=True, exist_ok=True)

    sorted_class_names = sorted(
        class_name_to_id.keys(), key=lambda x: class_name_to_id[x]
    )
    yaml.dump(
        {
            "path": str(folder.name),
            "train": "train/images",
            "val": "valid/images",
            "nc": len(class_name_to_id),
            "names": sorted_class_names,
        },
        (folder / "data.yaml").open("w"),
    )

    return folder


def merge_labeled_datasets(
    folders: list[str | Path], classes: list[str]
) -> LabelerDataset:
    """Merge a list of LabelerDatasets into a single LabelerDataset.

    Args:
        datasets (list[LabelerDataset]): The list of LabelerDatasets to merge.

    Returns:
        LabelerDataset: The merged LabelerDataset.
    """
    dataset = LabelerDataset(
        folder=folders[0],
        filename="classification.pickle",
        classes=classes,
    ).merge_datasets([LabelerDataset(folder) for folder in folders[1:]])

    return dataset


def copy_image(source_path: str | Path, destination_path: str | Path) -> None:
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    shutil.copy(source_path, destination_path)


def create_txt_file(
    detections: pd.DataFrame,
    destination_path: str | Path,
    class_name_to_id: dict,
) -> None:
    lines = []
    for _, row in detections.iterrows():
        if row["class_id"] is None:
            continue
        xywh = row["xywh"]
        class_id = row["class_id"]
        lines.append(f"{class_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}")

    destination_path = Path(destination_path)
    if destination_path.exists():
        destination_path.unlink()

    destination_path.write_text("\n".join(lines))


def add_group_to_dataset(
    image_path: Path | str,
    detections: pd.DataFrame,
    dataset_folder: str | Path,
    class_name_to_id: dict,
) -> pd.DataFrame:
    dataset_folder = Path(dataset_folder)
    image_path = Path(image_path)
    is_train = np.random.random() < 0.8

    if image_path.suffix not in [".jpg", ".jpeg", ".png"]:
        print(
            f"Skipping {image_path} for {dataset_folder} because it is not a valid image file"
        )
        return

    dataset = "train" if is_train else "valid"

    output_image_path = dataset_folder / dataset / "images" / image_path.name
    output_label_path = (
        dataset_folder / dataset / "labels" / image_path.with_suffix(".txt").name
    )

    copy_image(image_path, output_image_path)
    create_txt_file(detections, output_label_path, class_name_to_id)


class YoloV8Dataset:
    folder: Path
    df: pd.DataFrame
    classes: list[str]

    def __init__(self, folder: str | Path, classes: list[str], df: pd.DataFrame):
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
        create_dataset_folder(output_folder, self.class_name_to_id, delete_existing)
        for image_path, group in tqdm(self.df.groupby("image_path")):
            add_group_to_dataset(
                image_path,
                group,
                dataset_folder=output_folder,
                class_name_to_id=self.class_name_to_id,
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
            class_df = class_df.sample(
                min_samples, random_state=random_state
            ).reset_index(drop=True)
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
