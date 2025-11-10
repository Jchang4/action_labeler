"""Helper functions for dataset testing.

This module provides utility functions for creating test datasets,
asserting dataset equality, and other common test operations.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from action_labeler.datasets import YoloV8Dataset
from action_labeler.datasets.dataset_config import DatasetConfig


def create_temp_yolo_dataset(
    classes: list[str],
    num_images_per_split: dict[str, int] | None = None,
    detections_per_image: int = 3,
    random_seed: int = 42,
) -> tuple[Path, YoloV8Dataset]:
    """Create a temporary YOLOv8 dataset for testing.

    Args:
        classes: List of class names
        num_images_per_split: Dict mapping split names to number of images.
            Defaults to {"train": 10, "valid": 5}
        detections_per_image: Number of detections per image
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (temp_folder_path, YoloV8Dataset instance)
    """
    if num_images_per_split is None:
        num_images_per_split = {"train": 10, "valid": 5}

    rng = np.random.default_rng(random_seed)

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="yolo_test_"))

    # Create directory structure
    for split in num_images_per_split.keys():
        (temp_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (temp_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Create data.yaml
    with open(temp_dir / "data.yaml", "w") as f:
        yaml.dump(
            {
                "path": str(temp_dir.name),
                "train": "train/images",
                "val": "valid/images",
                "nc": len(classes),
                "names": classes,
            },
            f,
        )

    # Create synthetic images and labels
    data = []
    for split, num_images in num_images_per_split.items():
        for img_idx in range(num_images):
            # Create dummy image file (just a marker file, not a real image)
            image_path = temp_dir / split / "images" / f"image_{img_idx:04d}.jpg"
            image_path.write_text(f"dummy image {img_idx}")

            # Create labels
            label_path = temp_dir / split / "labels" / f"image_{img_idx:04d}.txt"
            label_lines = []

            for det_idx in range(detections_per_image):
                class_id = rng.integers(0, len(classes))
                # Generate random normalized bbox coordinates [x_center, y_center, width, height]
                x_center = rng.uniform(0.2, 0.8)
                y_center = rng.uniform(0.2, 0.8)
                width = rng.uniform(0.1, 0.3)
                height = rng.uniform(0.1, 0.3)

                label_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

                data.append(
                    {
                        "dataset": split,
                        "image_path": image_path,
                        "xywh": [x_center, y_center, width, height],
                        "class_id": class_id,
                    }
                )

            label_path.write_text("\n".join(label_lines))

    # Create dataset from the folder
    dataset = YoloV8Dataset.from_folder(temp_dir)

    return temp_dir, dataset


def cleanup_temp_dataset(temp_folder: Path) -> None:
    """Clean up a temporary dataset folder.

    Args:
        temp_folder: Path to the temporary folder to remove
    """
    if temp_folder.exists():
        shutil.rmtree(temp_folder)


def assert_dataset_equal(
    ds1: YoloV8Dataset,
    ds2: YoloV8Dataset,
    check_folder: bool = False,
    check_config: bool = False,
) -> None:
    """Assert that two datasets are equal.

    Args:
        ds1: First dataset
        ds2: Second dataset
        check_folder: Whether to check that folder paths match
        check_config: Whether to check that configs match

    Raises:
        AssertionError: If datasets are not equal
    """
    # Check classes
    assert (
        ds1.classes == ds2.classes
    ), f"Classes don't match: {ds1.classes} != {ds2.classes}"

    # Check class_name_to_id mapping
    assert ds1.class_name_to_id == ds2.class_name_to_id, (
        f"class_name_to_id doesn't match: "
        f"{ds1.class_name_to_id} != {ds2.class_name_to_id}"
    )

    # Check detection type
    assert ds1.detection_type == ds2.detection_type, (
        f"detection_type doesn't match: "
        f"{ds1.detection_type} != {ds2.detection_type}"
    )

    # Check dataframe equality
    assert len(ds1.df) == len(
        ds2.df
    ), f"DataFrame lengths don't match: {len(ds1.df)} != {len(ds2.df)}"

    # Sort dataframes for comparison (order may differ)
    df1_sorted = ds1.df.sort_values(
        by=["dataset", "image_path", "class_id"]
    ).reset_index(drop=True)
    df2_sorted = ds2.df.sort_values(
        by=["dataset", "image_path", "class_id"]
    ).reset_index(drop=True)

    # Compare dataframes column by column
    assert set(df1_sorted.columns) == set(df2_sorted.columns), (
        f"DataFrame columns don't match: "
        f"{df1_sorted.columns.tolist()} != {df2_sorted.columns.tolist()}"
    )

    if check_folder:
        assert (
            ds1.folder == ds2.folder
        ), f"Folders don't match: {ds1.folder} != {ds2.folder}"

    if check_config:
        assert (
            ds1.config == ds2.config
        ), f"Configs don't match: {ds1.config} != {ds2.config}"


def assert_class_distribution(
    dataset: YoloV8Dataset,
    expected_counts: dict[str, int],
    tolerance: int = 0,
) -> None:
    """Assert that the class distribution matches expectations.

    Args:
        dataset: Dataset to check
        expected_counts: Dict mapping class names to expected counts
        tolerance: Allowed deviation from expected counts

    Raises:
        AssertionError: If distribution doesn't match expectations
    """
    stats = dataset.stats
    actual_counts = stats.class_distribution

    for class_name, expected_count in expected_counts.items():
        actual_count = actual_counts.get(class_name, 0)
        assert abs(actual_count - expected_count) <= tolerance, (
            f"Class '{class_name}' count mismatch: "
            f"expected {expected_count} (±{tolerance}), got {actual_count}"
        )


def assert_split_ratio(
    dataset: YoloV8Dataset,
    expected_train_ratio: float,
    tolerance: float = 0.1,
) -> None:
    """Assert that the train/valid split ratio matches expectations.

    Args:
        dataset: Dataset to check
        expected_train_ratio: Expected ratio of training data (0.0 to 1.0)
        tolerance: Allowed deviation from expected ratio

    Raises:
        AssertionError: If split ratio doesn't match expectations
    """
    stats = dataset.stats
    total_images = stats.num_images
    if total_images == 0:
        return

    actual_train_ratio = stats.num_train_images / total_images
    assert abs(actual_train_ratio - expected_train_ratio) <= tolerance, (
        f"Train split ratio mismatch: "
        f"expected {expected_train_ratio} (±{tolerance}), "
        f"got {actual_train_ratio:.3f}"
    )


def create_empty_dataset_folder(temp_dir: Path, classes: list[str]) -> Path:
    """Create an empty dataset folder structure with data.yaml.

    Args:
        temp_dir: Temporary directory path
        classes: List of class names

    Returns:
        Path to the created folder
    """
    # Create directory structure
    for split in ["train", "valid"]:
        (temp_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (temp_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Create data.yaml
    with open(temp_dir / "data.yaml", "w") as f:
        yaml.dump(
            {
                "path": str(temp_dir.name),
                "train": "train/images",
                "val": "valid/images",
                "nc": len(classes),
                "names": classes,
            },
            f,
        )

    return temp_dir
