from pathlib import Path

import yaml


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
