"""I/O operations for YoloV8Dataset.

This module handles loading and saving YoloV8 datasets to/from disk.
"""

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from action_labeler.datasets.exceptions import DatasetIOError
from action_labeler.helpers.general import get_image_paths
from action_labeler.helpers.yolov8_dataset import (
    add_group_to_dataset_yolo_v8,
    create_dataset_folder,
    get_data_yaml,
    get_label_path,
    yolov8_labels_to_row,
)


class YoloV8DatasetIO:
    """Handles I/O operations for YoloV8Dataset."""

    @staticmethod
    def load_from_folder(folder: str | Path) -> tuple[Path, list[str], pd.DataFrame]:
        """Load a YoloV8 dataset from a folder.

        Args:
            folder: Path to the dataset folder containing data.yaml

        Returns:
            Tuple of (folder_path, classes, dataframe)

        Raises:
            DatasetIOError: If the dataset folder is invalid or cannot be loaded
        """
        folder = Path(folder)
        if not folder.exists():
            raise DatasetIOError(f"Dataset folder does not exist: {folder}")

        try:
            data_yaml = get_data_yaml(folder)
            classes = data_yaml["names"]
        except Exception as e:
            raise DatasetIOError(f"Failed to load data.yaml from {folder}: {e}")

        # Load images and text files
        data = []
        for dataset in ["train", "valid"]:
            dataset_path = folder / dataset / "images"
            if not dataset_path.exists():
                continue

            for image_path in get_image_paths(dataset_path):
                label_path = get_label_path(image_path)
                if label_path is None:
                    continue

                try:
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
                except Exception as e:
                    # Log but continue on individual file errors
                    print(f"Warning: Failed to load {label_path}: {e}")
                    continue

        df = pd.DataFrame(data)
        return folder, classes, df

    @staticmethod
    def save_to_folder(
        df: pd.DataFrame,
        output_folder: str | Path,
        class_name_to_id: dict[str, int],
        delete_existing: bool = False,
    ) -> None:
        """Save a YoloV8 dataset to a folder.

        Args:
            df: DataFrame containing the dataset
            output_folder: Path to save the dataset
            class_name_to_id: Mapping of class names to IDs
            delete_existing: Whether to delete existing folder

        Raises:
            DatasetIOError: If saving fails
        """
        try:
            # Add image_name column if not present
            df_copy = df.copy()
            if "image_name" not in df_copy.columns:
                df_copy["image_name"] = df_copy["image_path"].apply(
                    lambda x: Path(x).name
                )

            create_dataset_folder(output_folder, class_name_to_id, delete_existing)

            for image_path, group in tqdm(
                df_copy.groupby("image_path"), desc="Saving images"
            ):
                add_group_to_dataset_yolo_v8(
                    image_path,
                    group,
                    dataset_folder=output_folder,
                )
        except Exception as e:
            raise DatasetIOError(f"Failed to save dataset to {output_folder}: {e}")
