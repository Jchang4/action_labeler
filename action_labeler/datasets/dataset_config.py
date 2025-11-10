"""Configuration dataclasses for YoloV8Dataset.

This module defines configuration objects used throughout the dataset operations
to maintain consistency and provide sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Literal


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

    def __post_init__(self):
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
