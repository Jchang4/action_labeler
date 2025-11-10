"""YoloV8 Dataset management and manipulation.

This module provides a comprehensive interface for working with YOLOv8 format
object detection datasets, including loading, saving, transformation, validation,
and visualization capabilities.
"""

from pathlib import Path
from typing import Self

import pandas as pd

from action_labeler.dataclasses import DetectionType
from action_labeler.datasets.dataset_config import (
    DatasetConfig,
    DatasetStats,
    MergeStrategy,
    SplitType,
    ValidationResult,
)
from action_labeler.datasets.exceptions import (
    ClassNotFoundError,
    DatasetValidationError,
    InvalidSplitError,
)
from action_labeler.datasets.yolov8_dataset_io import YoloV8DatasetIO
from action_labeler.datasets.yolov8_dataset_transformer import YoloV8DatasetTransformer
from action_labeler.datasets.yolov8_dataset_validator import YoloV8DatasetValidator
from action_labeler.datasets.yolov8_dataset_visualizer import YoloV8DatasetVisualizer


class YoloV8Dataset:
    """YOLOv8 dataset management class.

    This class provides a comprehensive interface for working with YOLOv8 format
    datasets, including loading from disk, transforming (remapping/deleting classes,
    balancing), validating, visualizing, and saving.

    Attributes:
        folder: Path to the dataset folder
        df: DataFrame containing all detections with columns:
            - dataset: 'train' or 'valid'
            - image_path: Path to the image file
            - xywh: Bounding box in [x_center, y_center, width, height] format
            - class_id: Integer class ID
        classes: List of class names
        class_name_to_id: Mapping from class name to class ID
        detection_type: Type of detection (DETECT or SEGMENT)
        config: Dataset configuration

    Example:
        >>> # Load existing dataset
        >>> dataset = YoloV8Dataset.from_folder("path/to/dataset")
        >>>
        >>> # Create empty dataset
        >>> dataset = YoloV8Dataset.empty("output/path", ["dog", "cat"])
        >>>
        >>> # Remap classes
        >>> dataset.remap_classes({"dog": "canine", "cat": "feline"})
        >>>
        >>> # Balance dataset
        >>> balanced = dataset.create_balanced_dataset(min_samples=100)
        >>>
        >>> # Save dataset
        >>> dataset.save("output/path", delete_existing=True)
    """

    def __init__(
        self,
        folder: str | Path,
        classes: list[str],
        df: pd.DataFrame,
        detection_type: DetectionType = DetectionType.DETECT,
        config: DatasetConfig | None = None,
    ):
        """Initialize a YoloV8Dataset.

        Args:
            folder: Path to the dataset folder
            classes: List of class names
            df: DataFrame containing the dataset
            detection_type: Type of detection (DETECT or SEGMENT)
            config: Dataset configuration. If None, uses default config

        Raises:
            DatasetValidationError: If classes are invalid
        """
        # Validate inputs
        YoloV8DatasetValidator.validate_classes(classes)

        self.folder: Path = Path(folder)
        self.classes: list[str] = classes
        self.class_name_to_id: dict[str, int] = {
            class_name: i for i, class_name in enumerate(classes)
        }
        self.df: pd.DataFrame = df
        self.detection_type: DetectionType = detection_type
        self.config: DatasetConfig = config if config is not None else DatasetConfig()

        # Cache for expensive computations
        self._stats_cache: DatasetStats | None = None

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        detection_type: DetectionType = DetectionType.DETECT,
        config: DatasetConfig | None = None,
    ) -> Self:
        """Load a YoloV8 dataset from a folder.

        The folder should have the standard YOLOv8 structure:
        - folder/
            - train/
                - images/
                - labels/
            - valid/
                - images/
                - labels/
            - data.yaml

        Args:
            folder: Path to the dataset folder
            detection_type: Type of detection (DETECT or SEGMENT)
            config: Dataset configuration

        Returns:
            YoloV8Dataset instance

        Raises:
            DatasetIOError: If the folder structure is invalid or loading fails

        Example:
            >>> dataset = YoloV8Dataset.from_folder("./my_dataset")
            >>> print(f"Loaded {len(dataset)} detections")
        """
        folder_path, classes, df = YoloV8DatasetIO.load_from_folder(folder)
        return cls(folder_path, classes, df, detection_type, config)

    @classmethod
    def empty(
        cls,
        folder: str | Path,
        classes: list[str],
        detection_type: DetectionType = DetectionType.DETECT,
        config: DatasetConfig | None = None,
    ) -> Self:
        """Create an empty YoloV8 dataset.

        Args:
            folder: Path where the dataset will be saved
            classes: List of class names
            detection_type: Type of detection (DETECT or SEGMENT)
            config: Dataset configuration

        Returns:
            Empty YoloV8Dataset instance

        Example:
            >>> dataset = YoloV8Dataset.empty("./new_dataset", ["dog", "cat", "bird"])
        """
        folder_path = Path(folder)
        df = pd.DataFrame(columns=["dataset", "image_path", "xywh", "class_id"])
        return cls(folder_path, classes, df, detection_type, config)

    def save(self, output_folder: str | Path, delete_existing: bool = False) -> Self:
        """Save the dataset to disk in YOLOv8 format.

        Args:
            output_folder: Path to save the dataset
            delete_existing: Whether to delete existing folder before saving

        Returns:
            Self for method chaining

        Raises:
            DatasetIOError: If saving fails

        Example:
            >>> dataset.save("./output_dataset", delete_existing=True)
        """
        YoloV8DatasetIO.save_to_folder(
            self.df, output_folder, self.class_name_to_id, delete_existing
        )
        return self

    def remap_classes(self, old_to_new_class_name: dict[str, str]) -> Self:
        """Remap class names to new names (mutates in place).

        Args:
            old_to_new_class_name: Mapping from old class names to new names

        Returns:
            Self for method chaining

        Raises:
            ClassNotFoundError: If an old class name doesn't exist
            ClassMappingError: If remapping fails

        Example:
            >>> dataset.remap_classes({"dog": "canine", "cat": "feline"})
        """
        self.df, self.classes, self.class_name_to_id = (
            YoloV8DatasetTransformer.remap_classes(
                self.df, self.classes, self.class_name_to_id, old_to_new_class_name
            )
        )
        self._invalidate_cache()
        return self

    def delete_classes(self, class_names: list[str]) -> Self:
        """Delete specified classes from the dataset (mutates in place).

        Args:
            class_names: List of class names to delete

        Returns:
            Self for method chaining

        Raises:
            ClassNotFoundError: If a class doesn't exist

        Example:
            >>> dataset.delete_classes(["background", "unknown"])
        """
        self.df, self.classes, self.class_name_to_id = (
            YoloV8DatasetTransformer.delete_classes(
                self.df, self.classes, self.class_name_to_id, class_names
            )
        )
        self._invalidate_cache()
        return self

    def create_balanced_dataset(
        self, min_samples: int | None = None, random_state: int | None = 42
    ) -> Self:
        """Create a balanced dataset with equal samples per class.

        This method creates a NEW dataset instance with balanced class distribution.
        The original dataset is not modified.

        Args:
            min_samples: Minimum number of samples per class. If None, uses the
                minimum count across all classes
            random_state: Random seed for reproducibility

        Returns:
            New YoloV8Dataset instance with balanced classes

        Raises:
            EmptyDatasetError: If the dataset is empty

        Example:
            >>> balanced = dataset.create_balanced_dataset(min_samples=100)
            >>> balanced.plot_class_distribution()
        """
        # Create config with the specified random_state
        config = DatasetConfig(
            train_split=self.config.train_split,
            valid_split=self.config.valid_split,
            random_seed=random_state,
            allowed_image_formats=self.config.allowed_image_formats,
        )

        balanced_df = YoloV8DatasetTransformer.create_balanced_dataset(
            self.df, self.classes, min_samples, config
        )

        return YoloV8Dataset(
            self.folder,
            self.classes.copy(),
            balanced_df,
            self.detection_type,
            config,
        )

    def add_background_images(
        self, background_images_folder: str | Path, pct_background: float = 0.2
    ) -> Self:
        """Add background (negative) images to the dataset (mutates in place).

        Args:
            background_images_folder: Path to folder containing background images
            pct_background: Percentage of background images relative to min class count

        Returns:
            Self for method chaining

        Raises:
            EmptyDatasetError: If the dataset is empty

        Example:
            >>> dataset.add_background_images("./backgrounds", pct_background=0.15)
        """
        self.df = YoloV8DatasetTransformer.add_background_images(
            self.df, background_images_folder, pct_background, self.config
        )
        self._invalidate_cache()
        return self

    def merge(self, other: Self, strategy: MergeStrategy = "union") -> Self:
        """Merge another dataset into this one.

        Args:
            other: Another YoloV8Dataset to merge
            strategy: Merge strategy - 'union' (keep all classes) or
                'intersection' (only common classes)

        Returns:
            New merged YoloV8Dataset instance

        Raises:
            DatasetValidationError: If datasets are incompatible

        Example:
            >>> dataset1 = YoloV8Dataset.from_folder("dataset1")
            >>> dataset2 = YoloV8Dataset.from_folder("dataset2")
            >>> merged = dataset1.merge(dataset2, strategy="union")
        """
        if strategy == "union":
            # Union: combine all classes from both datasets
            new_classes = self.classes.copy()
            for class_name in other.classes:
                if class_name not in new_classes:
                    new_classes.append(class_name)

            new_class_name_to_id = {name: idx for idx, name in enumerate(new_classes)}

            # Remap class IDs in both datasets
            df1 = self.df.copy()
            df1["class_id"] = df1["class_id"].apply(
                lambda x: (
                    new_class_name_to_id[self.classes[int(x)]] if pd.notna(x) else x
                )
            )

            df2 = other.df.copy()
            df2["class_id"] = df2["class_id"].apply(
                lambda x: (
                    new_class_name_to_id[other.classes[int(x)]] if pd.notna(x) else x
                )
            )

            # Combine dataframes
            merged_df = pd.concat([df1, df2], ignore_index=True)

        else:  # intersection
            # Intersection: only keep classes that exist in both datasets
            common_classes = [c for c in self.classes if c in other.classes]

            if not common_classes:
                raise DatasetValidationError(
                    "Cannot merge datasets with intersection strategy: "
                    "no common classes found"
                )

            new_class_name_to_id = {
                name: idx for idx, name in enumerate(common_classes)
            }

            # Filter and remap class IDs
            df1 = self.df[
                self.df["class_id"].apply(
                    lambda x: pd.isna(x) or self.classes[int(x)] in common_classes
                )
            ].copy()
            df1["class_id"] = df1["class_id"].apply(
                lambda x: (
                    new_class_name_to_id[self.classes[int(x)]] if pd.notna(x) else x
                )
            )

            df2 = other.df[
                other.df["class_id"].apply(
                    lambda x: pd.isna(x) or other.classes[int(x)] in common_classes
                )
            ].copy()
            df2["class_id"] = df2["class_id"].apply(
                lambda x: (
                    new_class_name_to_id[other.classes[int(x)]] if pd.notna(x) else x
                )
            )

            merged_df = pd.concat([df1, df2], ignore_index=True)
            new_classes = common_classes

        # Use self.folder as the folder for the merged dataset
        return YoloV8Dataset(
            self.folder,
            new_classes,
            merged_df,
            self.detection_type,
            self.config,
        )

    def filter_by_split(self, split: SplitType) -> Self:
        """Filter dataset to a specific split (train or valid).

        Args:
            split: Split to filter to ('train' or 'valid')

        Returns:
            New filtered YoloV8Dataset instance

        Raises:
            InvalidSplitError: If split is not valid

        Example:
            >>> train_dataset = dataset.filter_by_split("train")
        """
        valid_splits = ["train", "valid"]
        if split not in valid_splits:
            raise InvalidSplitError(split, valid_splits)

        filtered_df = self.df[self.df["dataset"] == split].copy()

        return YoloV8Dataset(
            self.folder,
            self.classes.copy(),
            filtered_df,
            self.detection_type,
            self.config,
        )

    def filter_by_classes(self, class_names: list[str]) -> Self:
        """Filter dataset to only include specified classes.

        Args:
            class_names: List of class names to keep

        Returns:
            New filtered YoloV8Dataset instance

        Raises:
            ClassNotFoundError: If a class doesn't exist

        Example:
            >>> subset = dataset.filter_by_classes(["dog", "cat"])
        """
        # Verify all classes exist
        for class_name in class_names:
            if class_name not in self.class_name_to_id:
                raise ClassNotFoundError(class_name, self.classes)

        # Get class IDs to keep
        class_ids_to_keep = [self.class_name_to_id[name] for name in class_names]

        # Filter dataframe (keep background images too)
        filtered_df = self.df[
            self.df["class_id"].isna() | self.df["class_id"].isin(class_ids_to_keep)
        ].copy()

        # Create new class list and remap IDs
        new_class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
        old_to_new_id = {
            self.class_name_to_id[name]: new_class_name_to_id[name]
            for name in class_names
        }

        filtered_df["class_id"] = filtered_df["class_id"].map(old_to_new_id)

        return YoloV8Dataset(
            self.folder,
            class_names,
            filtered_df,
            self.detection_type,
            self.config,
        )

    def validate(self, check_files_exist: bool = True) -> ValidationResult:
        """Validate the dataset for integrity and correctness.

        Args:
            check_files_exist: Whether to verify that image files exist on disk

        Returns:
            ValidationResult with validation status and any errors/warnings

        Example:
            >>> result = dataset.validate()
            >>> print(result)
            >>> if not result.is_valid:
            ...     print("Validation failed!")
        """
        return YoloV8DatasetValidator.validate_dataframe(
            self.df, self.classes, check_files_exist
        )

    @property
    def stats(self) -> DatasetStats:
        """Get comprehensive statistics about the dataset.

        Returns:
            DatasetStats object with detailed statistics

        Example:
            >>> stats = dataset.stats
            >>> print(f"Total images: {stats.num_images}")
            >>> print(stats)  # Print full statistics
        """
        if self._stats_cache is not None:
            return self._stats_cache

        # Calculate statistics
        num_images = self.df["image_path"].nunique()
        num_detections = len(self.df)
        num_classes = len(self.classes)

        # Split statistics
        train_df = self.df[self.df["dataset"] == "train"]
        valid_df = self.df[self.df["dataset"] == "valid"]

        num_train_images = train_df["image_path"].nunique()
        num_valid_images = valid_df["image_path"].nunique()
        num_train_detections = len(train_df)
        num_valid_detections = len(valid_df)

        # Class distribution
        class_distribution = {}
        images_per_class = {}

        for class_id, class_name in enumerate(self.classes):
            class_df = self.df[self.df["class_id"] == class_id]
            class_distribution[class_name] = len(class_df)
            images_per_class[class_name] = class_df["image_path"].nunique()

        # Calculate averages and ranges
        avg_detections_per_image = (
            num_detections / num_images if num_images > 0 else 0.0
        )

        detection_counts = list(class_distribution.values())
        min_detections = min(detection_counts) if detection_counts else 0
        max_detections = max(detection_counts) if detection_counts else 0

        self._stats_cache = DatasetStats(
            num_images=num_images,
            num_detections=num_detections,
            num_classes=num_classes,
            num_train_images=num_train_images,
            num_valid_images=num_valid_images,
            num_train_detections=num_train_detections,
            num_valid_detections=num_valid_detections,
            class_distribution=class_distribution,
            images_per_class=images_per_class,
            avg_detections_per_image=avg_detections_per_image,
            min_detections_per_class=min_detections,
            max_detections_per_class=max_detections,
        )

        return self._stats_cache

    def plot_class_distribution(self, title: str | None = None) -> Self:
        """Plot the distribution of classes across train and validation sets.

        Args:
            title: Optional custom title for the plot

        Returns:
            Self for method chaining

        Example:
            >>> dataset.plot_class_distribution()
        """
        YoloV8DatasetVisualizer.plot_class_distribution(self.df, self.classes, title)
        return self

    def plot_split_distribution(self) -> Self:
        """Plot the distribution of train/valid splits.

        Returns:
            Self for method chaining

        Example:
            >>> dataset.plot_split_distribution()
        """
        YoloV8DatasetVisualizer.plot_split_distribution(self.df)
        return self

    def plot_detections_per_image(self) -> Self:
        """Plot histogram of number of detections per image.

        Returns:
            Self for method chaining

        Example:
            >>> dataset.plot_detections_per_image()
        """
        YoloV8DatasetVisualizer.plot_detections_per_image(self.df)
        return self

    def plot_bbox_size_distribution(self) -> Self:
        """Plot distribution of bounding box sizes.

        Returns:
            Self for method chaining

        Example:
            >>> dataset.plot_bbox_size_distribution()
        """
        YoloV8DatasetVisualizer.plot_bbox_size_distribution(self.df)
        return self

    def plot_dataset(self) -> Self:
        """Plot comprehensive dataset visualizations.

        This is a convenience method that creates multiple plots to give
        a complete overview of the dataset.

        Returns:
            Self for method chaining

        Example:
            >>> dataset.plot_dataset()
        """
        self.plot_class_distribution()
        self.plot_split_distribution()
        self.plot_detections_per_image()
        self.plot_bbox_size_distribution()
        return self

    def plot_class(self, class_name: str, num_samples: int = 5) -> Self:
        """Plot sample images for a specific class.

        Args:
            class_name: Name of the class to visualize
            num_samples: Number of sample images to show

        Returns:
            Self for method chaining

        Raises:
            ClassNotFoundError: If the class doesn't exist

        Example:
            >>> dataset.plot_class("dog", num_samples=10)
        """
        if class_name not in self.class_name_to_id:
            raise ClassNotFoundError(class_name, self.classes)

        class_id = self.class_name_to_id[class_name]
        YoloV8DatasetVisualizer.plot_class_samples(
            self.df, self.classes, class_name, class_id, num_samples
        )
        return self

    def copy(self) -> Self:
        """Create a deep copy of the dataset.

        Returns:
            New YoloV8Dataset instance with copied data

        Example:
            >>> dataset_copy = dataset.copy()
            >>> dataset_copy.delete_classes(["background"])  # Doesn't affect original
        """
        return YoloV8Dataset(
            self.folder,
            self.classes.copy(),
            self.df.copy(),
            self.detection_type,
            self.config,
        )

    def _invalidate_cache(self) -> None:
        """Invalidate cached computed properties."""
        self._stats_cache = None

    def __len__(self) -> int:
        """Return the number of detections in the dataset.

        Returns:
            Number of detections

        Example:
            >>> print(f"Dataset has {len(dataset)} detections")
        """
        return len(self.df)

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            String representation

        Example:
            >>> dataset
            YoloV8Dataset(classes=3, images=150, detections=500, splits=['train', 'valid'])
        """
        num_images = self.df["image_path"].nunique() if len(self.df) > 0 else 0
        splits = sorted(self.df["dataset"].unique()) if len(self.df) > 0 else []

        return (
            f"YoloV8Dataset("
            f"classes={len(self.classes)}, "
            f"images={num_images}, "
            f"detections={len(self.df)}, "
            f"splits={splits}"
            f")"
        )
