import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from action_labeler.dataclasses import DetectionType


def get_data_yaml(dataset_path: Path | str, verbose: bool = False) -> dict:
    """Get the data.yaml file from a dataset path.

    Args:
        dataset_path (Path | str): The path to the dataset.
        verbose (bool, optional): Whether to print the data.yaml file. Defaults to False.

    Returns:
        dict: The data.yaml file.
    """
    dataset_path = Path(dataset_path)
    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data YAML file not found at {data_yaml_path}")

    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    if verbose:
        print(f"Dataset path: {dataset_path}")
        print("Class Names:")
        for i, class_name in enumerate(data_yaml["names"]):
            print(f"{i}: {class_name}")
        print(f"Number of classes: {len(data_yaml['names'])}")

    return data_yaml


def get_label_path(image_path: str | Path) -> Path | None:
    image_path = Path(image_path)
    text_path = (
        image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name
    )
    if not text_path.exists():
        return None
    return text_path


def yolov8_labels_to_row(label_path: Path | str) -> dict:
    label_path = Path(label_path)
    return [
        [float(num) for num in line.split(" ")]
        for line in label_path.read_text().splitlines()
        if line and len(line.split(" ")) > 1
    ]


def ultralytics_labels_to_xywh(txt_path: Path | str) -> list[list[float]]:
    """Convert a Ultralytics labels txt file to a list of xywh boxes.

    Ultralytics labels formats: https://docs.ultralytics.com/modes/predict/#working-with-results

    Note: this method can also handle segmentation and keypoint detection.

    Args:
        txt_path (Path | str): The path to the txt file.

    Returns:
        list[list[float]]: A list of xywh boxes.
    """
    return [row[1:] for row in yolov8_labels_to_row(txt_path)]


# ------------------------------------------------------------------------------------------------ #
# Create Yolo V8 Dataset Helpers
# ------------------------------------------------------------------------------------------------ #
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


def copy_image(source_path: str | Path, destination_path: str | Path) -> None:
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    shutil.copy(source_path, destination_path)


def create_txt_file(
    detections: pd.DataFrame,
    destination_path: str | Path,
    detection_type: DetectionType,
) -> None:
    lines = []
    for _, row in detections.iterrows():
        if row["class_id"] is None:
            continue

        if detection_type == DetectionType.DETECT:
            xywh = row["xywh"]
        elif detection_type == DetectionType.SEGMENT:
            xywh = row["segmentation_points"]

        class_id = row["class_id"]
        lines.append(f"{class_id} {' '.join(map(str, xywh))}")

    destination_path = Path(destination_path)
    if destination_path.exists():
        destination_path.unlink()

    destination_path.write_text("\n".join(lines))


def add_group_to_dataset(
    image_path: Path | str,
    detections: pd.DataFrame,
    dataset_folder: str | Path,
    detection_type: DetectionType,
) -> pd.DataFrame:
    assert sorted(detections.keys()) == sorted(
        ["image_path", "xywh", "segmentation_points", "action", "class_id"]
    ), f"The detections dataframe must have the columns 'image_path', 'xywh', 'segmentation_points', 'action', and 'class_id'. Got {sorted(detections.keys())}"

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
    create_txt_file(detections, output_label_path, detection_type)


def add_group_to_dataset_yolo_v8(
    image_path: Path | str,
    detections: pd.DataFrame,
    dataset_folder: str | Path,
) -> pd.DataFrame:
    assert sorted(detections.keys()) == sorted(
        ["class_id", "dataset", "image_name", "image_path", "xywh"]
    ), f"The detections dataframe must have the columns 'class_id', 'dataset', 'image_name', 'image_path', and 'xywh'. Got {sorted(detections.keys())}"

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
    create_txt_file(detections, output_label_path, DetectionType.DETECT)
