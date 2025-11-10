"""Pytest fixtures for dataset tests.

This module provides pytest fixtures for creating test datasets
and other common test resources.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from tests.datasets.helpers import cleanup_temp_dataset, create_temp_yolo_dataset

from action_labeler.datasets import YoloV8Dataset
from action_labeler.datasets.dataset_config import DatasetConfig


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp(prefix="yolo_test_"))
    yield temp_path
    cleanup_temp_dataset(temp_path)


@pytest.fixture
def sample_classes():
    """Provide a sample list of class names."""
    return ["dog", "cat", "bird", "fish"]


@pytest.fixture
def sample_dataset(sample_classes):
    """Provide a sample YoloV8Dataset with synthetic data."""
    temp_folder, dataset = create_temp_yolo_dataset(
        classes=sample_classes,
        num_images_per_split={"train": 10, "valid": 5},
        detections_per_image=3,
        random_seed=42,
    )
    yield dataset
    cleanup_temp_dataset(temp_folder)


@pytest.fixture
def empty_dataset(temp_dir, sample_classes):
    """Provide an empty YoloV8Dataset."""
    return YoloV8Dataset.empty(temp_dir, sample_classes)


@pytest.fixture
def multi_class_dataset():
    """Provide a dataset with many classes for testing balancing."""
    classes = [f"class_{i}" for i in range(10)]
    temp_folder, dataset = create_temp_yolo_dataset(
        classes=classes,
        num_images_per_split={"train": 20, "valid": 10},
        detections_per_image=5,
        random_seed=42,
    )
    yield dataset
    cleanup_temp_dataset(temp_folder)


@pytest.fixture
def unbalanced_dataset(temp_dir):
    """Provide an unbalanced dataset for testing balancing operations."""
    classes = ["common", "rare", "very_rare"]

    # Manually create unbalanced data
    data = []

    # Common class: 100 samples
    for i in range(100):
        data.append(
            {
                "dataset": "train" if i < 80 else "valid",
                "image_path": temp_dir / f"image_common_{i}.jpg",
                "xywh": [0.5, 0.5, 0.3, 0.3],
                "class_id": 0,
            }
        )

    # Rare class: 30 samples
    for i in range(30):
        data.append(
            {
                "dataset": "train" if i < 24 else "valid",
                "image_path": temp_dir / f"image_rare_{i}.jpg",
                "xywh": [0.5, 0.5, 0.3, 0.3],
                "class_id": 1,
            }
        )

    # Very rare class: 10 samples
    for i in range(10):
        data.append(
            {
                "dataset": "train" if i < 8 else "valid",
                "image_path": temp_dir / f"image_very_rare_{i}.jpg",
                "xywh": [0.5, 0.5, 0.3, 0.3],
                "class_id": 2,
            }
        )

    df = pd.DataFrame(data)
    return YoloV8Dataset(temp_dir, classes, df)


@pytest.fixture
def dataset_config():
    """Provide a default DatasetConfig."""
    return DatasetConfig()


@pytest.fixture
def custom_config():
    """Provide a custom DatasetConfig with non-default values."""
    return DatasetConfig(train_split=0.7, valid_split=0.3, random_seed=123)


@pytest.fixture
def two_class_dataset():
    """Provide a simple two-class dataset for basic testing."""
    classes = ["positive", "negative"]
    temp_folder, dataset = create_temp_yolo_dataset(
        classes=classes,
        num_images_per_split={"train": 5, "valid": 3},
        detections_per_image=2,
        random_seed=42,
    )
    yield dataset
    cleanup_temp_dataset(temp_folder)
