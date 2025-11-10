"""Configuration dataclasses for YoloV8Dataset.

This module defines configuration objects used throughout the dataset operations
to maintain consistency and provide sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset operations.

    Attributes:
        train_split: Proportion of data to use for training (0.0 to 1.0)
        valid_split: Proportion of data to use for validation (0.0 to 1.0)
        random_seed: Random seed for reproducibility. None for non-deterministic behavior
        allowed_image_formats: Tuple of allowed image file extensions

    Example:
        >>> config = DatasetConfig(train_split=0.7, valid_split=0.3)
        >>> config = DatasetConfig(random_seed=None)  # Non-deterministic
    """

    train_split: float = 0.8
    valid_split: float = 0.2
    random_seed: int | None = 42
    allowed_image_formats: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.train_split <= 1.0:
            raise ValueError(
                f"train_split must be between 0 and 1, got {self.train_split}"
            )
        if not 0.0 <= self.valid_split <= 1.0:
            raise ValueError(
                f"valid_split must be between 0 and 1, got {self.valid_split}"
            )
        if abs(self.train_split + self.valid_split - 1.0) > 1e-6:
            raise ValueError(
                f"train_split + valid_split must equal 1.0, "
                f"got {self.train_split} + {self.valid_split} = "
                f"{self.train_split + self.valid_split}"
            )

    def get_rng(self) -> np.random.Generator:
        """Get a random number generator with the configured seed.

        Returns:
            numpy random Generator instance

        Example:
            >>> config = DatasetConfig(random_seed=42)
            >>> rng = config.get_rng()
            >>> random_values = rng.random(10)
        """
        return (
            np.random.default_rng(self.random_seed)
            if self.random_seed is not None
            else np.random.default_rng()
        )


@dataclass
class DatasetStats:
    """Statistics about a YoloV8Dataset.

    Attributes:
        num_images: Total number of unique images
        num_detections: Total number of object detections
        num_classes: Number of unique classes
        num_train_images: Number of images in training set
        num_valid_images: Number of images in validation set
        num_train_detections: Number of detections in training set
        num_valid_detections: Number of detections in validation set
        class_distribution: Dict mapping class names to detection counts
        images_per_class: Dict mapping class names to unique image counts
        avg_detections_per_image: Average number of detections per image
        min_detections_per_class: Minimum number of detections across all classes
        max_detections_per_class: Maximum number of detections across all classes
    """

    num_images: int
    num_detections: int
    num_classes: int
    num_train_images: int
    num_valid_images: int
    num_train_detections: int
    num_valid_detections: int
    class_distribution: dict[str, int]
    images_per_class: dict[str, int]
    avg_detections_per_image: float
    min_detections_per_class: int
    max_detections_per_class: int

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        lines = [
            "Dataset Statistics:",
            f"  Total Images: {self.num_images} (Train: {self.num_train_images}, Valid: {self.num_valid_images})",
            f"  Total Detections: {self.num_detections} (Train: {self.num_train_detections}, Valid: {self.num_valid_detections})",
            f"  Number of Classes: {self.num_classes}",
            f"  Avg Detections/Image: {self.avg_detections_per_image:.2f}",
            f"  Class Distribution Range: {self.min_detections_per_class} - {self.max_detections_per_class}",
            "",
            "Class Distribution:",
        ]
        for class_name, count in sorted(
            self.class_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {class_name}: {count}")
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Result of dataset validation.

    Attributes:
        is_valid: Whether the dataset passed all validation checks
        errors: List of validation error messages
        warnings: List of validation warning messages
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        if self.is_valid and not self.warnings:
            return "✓ Dataset validation passed"

        lines = []
        if not self.is_valid:
            lines.append("✗ Dataset validation failed")
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  - {error}")
        else:
            lines.append("✓ Dataset validation passed with warnings")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


MergeStrategy = Literal["union", "intersection"]
SplitType = Literal["train", "valid"]


def map_class_id_to_name(
    class_id: int | float | None, classes: list[str], default: str = "background"
) -> str:
    """Map a class ID to its corresponding class name.

    Args:
        class_id: The class ID to map (can be None for background images)
        classes: List of class names
        default: Default value to return if class_id is None

    Returns:
        The class name corresponding to the ID, or default if None

    Example:
        >>> classes = ["dog", "cat", "bird"]
        >>> map_class_id_to_name(0, classes)
        'dog'
        >>> map_class_id_to_name(None, classes)
        'background'
    """
    return classes[int(class_id)] if pd.notna(class_id) else default


def create_class_id_mapping(
    class_id: int | float | None, class_name_to_id: dict[str, int]
) -> int | None:
    """Create a mapping function for class IDs during transformations.

    This is used when remapping class IDs during merge or transformation operations.

    Args:
        class_id: The original class ID
        class_name_to_id: Mapping from class names to new IDs

    Returns:
        The mapped class ID, or None if input is None

    Example:
        >>> mapping = {"dog": 0, "cat": 1}
        >>> create_class_id_mapping(0, mapping)
        0
    """
    return class_name_to_id.get(int(class_id)) if pd.notna(class_id) else None
