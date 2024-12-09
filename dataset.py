import shutil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm.auto import tqdm

from .helpers import get_detection, xywh_to_xyxy, xyxy_to_mask
from .preprocessors import BoxImagePreprocessor


class Dataset:
    df: pd.DataFrame
    classes: list[str]

    def __init__(self, folder: str | Path):
        self.df = self._load_df(Path(folder))
        self._remove_empty_classes()

    def _load_df(self, folder: Path) -> pd.DataFrame:
        config = yaml.load((folder / "data.yaml").read_text(), Loader=yaml.FullLoader)
        self.classes = config["names"]

        data = []

        for dataset in ["train", "valid"]:
            images_path = folder / dataset / "images"
            labels_path = folder / dataset / "labels"

            for image_path in images_path.iterdir():
                # Skip missing label txt file
                label_path = labels_path / f"{image_path.stem}.txt"
                if not label_path.exists():
                    print(f"Skipping {image_path.stem}")
                    continue

                raw_labels = [
                    line.strip().split(" ")
                    for line in label_path.read_text().splitlines()
                    if line
                ]

                for raw_label in raw_labels:
                    class_id = int(raw_label[0])
                    xywh = " ".join(raw_label[1:])
                    data.append(
                        {
                            "dataset": dataset,
                            "image_path": str(image_path),
                            "class_name": self.classes[class_id],
                            "class_id": class_id,
                            "xywh": xywh,
                        }
                    )

        return pd.DataFrame(data)

    def combine_datasets(self, datasets: list["Dataset"]) -> "Dataset":
        for next_dataset in datasets:
            # Combine classes
            combined_classes = sorted(set(self.classes) | set(next_dataset.classes))
            curr_ds_class_map = {c: combined_classes.index(c) for c in self.classes}
            next_ds_class_map = {
                c: combined_classes.index(c) for c in next_dataset.classes
            }

            # Update class ids
            self.df["class_id"] = self.df["class_name"].map(curr_ds_class_map)
            next_dataset.df["class_id"] = next_dataset.df["class_name"].map(
                next_ds_class_map
            )

            # Combine dataframes
            # If the image_path name is in both datasets, drop the curr and use labels in next_dataset
            self.df["image_name"] = self.df["image_path"].apply(lambda x: Path(x).name)
            next_dataset.df["image_name"] = next_dataset.df["image_path"].apply(
                lambda x: Path(x).name
            )

            self.df = pd.concat(
                [
                    self.df[~self.df["image_name"].isin(next_dataset.df["image_name"])],
                    next_dataset.df,
                ]
            ).reset_index(drop=True)

            # Drop the temporary 'image_name' column
            self.df.drop(columns=["image_name"], inplace=True)
            self.classes = combined_classes

        self._remove_empty_classes()

        return self

    def save(self, output_folder: str | Path):
        output_folder = Path(output_folder)
        _remake_dataset_dir(output_folder)

        # Save data.yaml
        data = {
            "train": "train/images",
            "val": "valid/images",
            "path": output_folder.name,
            "names": self.classes,
        }
        yaml.dump(data, (output_folder / "data.yaml").open("w"))

        # Save images and labels
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_path = (
                output_folder / row["dataset"] / "images" / Path(row["image_path"]).name
            )
            label_path = (
                output_folder
                / row["dataset"]
                / "labels"
                / Path(row["image_path"]).with_suffix(".txt").name
            )

            shutil.copy(row["image_path"], image_path)
            label = f"{row['class_id']} {row['xywh']}"
            with label_path.open("a") as f:
                f.write(label + "\n")

    def remap_classes(self, class_map: dict[str, str]) -> "Dataset":
        """Remap classes in the dataset

        Args:
            class_map (dict[str, str]): mapping of old class names to new class names
        """
        # Ensure all classes exists in self.classes
        self._assert_classes_exist(class_map.keys())
        # Ensure old and new classes do not overlap
        assert set(class_map.keys()) & set(class_map.values()) == set(), (
            "Old and new classes cannot overlap. "
            f"Overlapping classes: {set(class_map.keys()) & set(class_map.values())}"
        )

        new_classes = sorted(
            set(self.classes) - set(class_map.keys()) | set(class_map.values())
        )
        old_to_new_class_id = {
            c: new_classes.index(class_map.get(c, c)) for c in self.classes
        }
        old_to_new_class_name = {c: class_map.get(c, c) for c in self.classes}

        # Update class ids
        self.df["class_id"] = self.df["class_name"].map(old_to_new_class_id)
        self.df["class_name"] = self.df["class_name"].map(old_to_new_class_name)
        self.classes = new_classes

        self._remove_empty_classes()

        return self

    def delete_classes(self, classes: list[str]) -> "Dataset":
        """Delete classes from the dataset

        Args:
            classes (list[str]): list of classes to delete
        """
        # Ensure all classes exists in self.classes
        self._assert_classes_exist(classes)

        # Filter out classes
        self.df = self.df[~self.df["class_name"].isin(classes)].reset_index(drop=True)

        # Class remap to remove gaps in class ids
        new_classes = sorted(set(self.classes) - set(classes))
        old_to_new_class_id = {c: new_classes.index(c) for c in new_classes}
        old_to_new_class_name = {c: c for c in new_classes}

        # Update class ids
        self.df["class_name"] = self.df["class_name"].map(old_to_new_class_name)
        self.df["class_id"] = self.df["class_name"].map(old_to_new_class_id)
        self.classes = new_classes

        self._remove_empty_classes()

        return self

    def sample_by_prefix(self, prefix: str, num_samples: int = 50) -> "Dataset":
        """For image_paths containing the prefix, sample `num_samples` images for each class"""
        prefix_df = self.df[self.df["image_path"].str.contains(prefix)]

        # Sample `num_samples` images for each class
        sampled_df = prefix_df.groupby("class_name").apply(
            lambda x: x.sample(min(num_samples, len(x)))
        )
        # Remove prefix_df from self.df and add sampled_df back
        self.df = self.df[~self.df["image_path"].str.contains(prefix)].reset_index(
            drop=True
        )
        self.df = pd.concat([self.df, sampled_df]).reset_index(drop=True)

        return self

    def get_balanced_dataset(self, num_samples: Optional[int] = None) -> "Dataset":
        """Get a balanced dataset with `num_samples` samples for each class"""
        if num_samples is None:
            num_samples = self.df["class_name"].value_counts().min()

        balanced_df = (
            self.df.groupby("class_name")
            .apply(lambda x: x.sample(min(num_samples, len(x))))
            .reset_index(drop=True)
        )

        # Randomly set the dataset to train or valid
        balanced_df["dataset"] = balanced_df["dataset"].apply(
            lambda x: "train" if np.random.rand() < 0.8 else "valid"
        )

        self.df = balanced_df

        return self

    def plot_class_distribution(self) -> "Dataset":
        """Create two bar plots showing the class distribution in the train and valid datasets.
        Graphs are plotted vertically.
        """
        _, ax = plt.subplots(2, 1, figsize=(12, 12))
        for i, dataset in enumerate(["train", "valid"]):
            class_counts = self.df[self.df["dataset"] == dataset][
                "class_name"
            ].value_counts()

            print(f"Dataset: {dataset}")
            print(class_counts, "\n")

            class_counts.plot(kind="bar", ax=ax[i])
            ax[i].set_title(f"{dataset.capitalize()} Dataset")
            ax[i].set_ylabel("Count")
            ax[i].set_xlabel("Class Name")
            ax[i].grid(axis="y")
        plt.tight_layout()
        plt.show()

        return self

    def plot_dataset(self, num_images: int = 5) -> "Dataset":
        """Plot N images for each class from the dataset with bounding boxes"""
        box_annotator = BoxImagePreprocessor()

        for class_name in self.classes:
            class_df = self.df[self.df["class_name"] == class_name]
            class_df = class_df.sample(min(num_images, len(class_df)))

            if len(class_df) == 0:
                print(f"No images found for class: {class_name}")
                continue

            fig, axes = plt.subplots(1, len(class_df), figsize=(12, 5))

            if len(class_df) == 1:
                axes = [axes]  # Make it iterable if there's only one image

            for ax, (_, row) in zip(axes, class_df.iterrows()):
                image = Image.open(row["image_path"])
                image.thumbnail((480, 480))

                xyxys = [xywh_to_xyxy(image, list(map(float, row["xywh"].split(" "))))]
                masks = [xyxy_to_mask(image, xyxy) for xyxy in xyxys]
                detections = get_detection(xyxys, masks)
                image = box_annotator.preprocess(image, 0, detections)

                ax.imshow(image)
                ax.axis("off")

            plt.suptitle(class_name)
            plt.show()

        return self

    def plot_class(self, class_name: str, num_images: int = 20) -> "Dataset":
        class_df = self.df[self.df["class_name"] == class_name]
        class_df = class_df.sample(min(num_images, len(class_df)))

        fig, axes = plt.subplots(num_images // 2, 2, figsize=(20, 20))

        for ax, (_, row) in zip(axes.ravel(), class_df.iterrows()):
            image = Image.open(row["image_path"])
            image.thumbnail((480, 480))
            ax.imshow(image)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        return self

    def _assert_classes_exist(self, classes: list[str]):
        missing_classes = set(classes) - set(self.classes)
        assert (
            not missing_classes
        ), f"Classes do not exist in the dataset: {missing_classes}"

    def _remove_empty_classes(self):
        empty_classes = set(self.classes) - set(self.df["class_name"].unique())
        if not empty_classes:
            return
        self.delete_classes(list(empty_classes))


def _remake_dataset_dir(folder: Path):
    folder.mkdir(exist_ok=True, parents=True)

    # Remove existing train and valid folders, data.yaml
    (folder / "data.yaml").unlink(missing_ok=True)
    shutil.rmtree(folder / "train", ignore_errors=True)
    shutil.rmtree(folder / "valid", ignore_errors=True)

    # Create new train and valid folders
    (folder / "train" / "images").mkdir(parents=True, exist_ok=True)
    (folder / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (folder / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (folder / "valid" / "labels").mkdir(parents=True, exist_ok=True)
