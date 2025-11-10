"""Transformation operations for YoloV8Dataset.

This module handles transformations like class remapping, deletion, balancing, etc.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from action_labeler.datasets.dataset_config import DatasetConfig
from action_labeler.datasets.exceptions import (
    ClassMappingError,
    ClassNotFoundError,
    EmptyDatasetError,
)
from action_labeler.helpers.general import get_image_paths


class YoloV8DatasetTransformer:
    """Handles transformation operations for YoloV8Dataset."""

    @staticmethod
    def remap_classes(
        df: pd.DataFrame,
        classes: list[str],
        class_name_to_id: dict[str, int],
        old_to_new_class_name: dict[str, str],
    ) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
        """Remap class names to new names.

        Args:
            df: DataFrame containing the dataset
            classes: Current list of class names
            class_name_to_id: Current class name to ID mapping
            old_to_new_class_name: Mapping from old class names to new class names

        Returns:
            Tuple of (new_df, new_classes, new_class_name_to_id)

        Raises:
            ClassNotFoundError: If an old class name doesn't exist
            ClassMappingError: If remapping fails
        """
        df_copy = df.copy()
        new_classes = classes.copy()
        new_class_name_to_id = class_name_to_id.copy()

        # Build complete mapping (including unmapped classes)
        old_to_new_class_id = {}

        # First, handle classes being remapped
        for old_class_name, new_class_name in tqdm(
            old_to_new_class_name.items(), desc="Remapping classes"
        ):
            # Verify old class exists
            if old_class_name not in class_name_to_id:
                raise ClassNotFoundError(old_class_name, classes)

            old_class_id = class_name_to_id[old_class_name]

            # If new class name doesn't exist, add it and map to old ID
            if new_class_name not in new_class_name_to_id:
                # Replace old class name with new class name at the same position
                old_idx = new_classes.index(old_class_name)
                new_classes[old_idx] = new_class_name
                # Update mappings
                new_class_name_to_id[new_class_name] = old_class_id
                del new_class_name_to_id[old_class_name]
                new_class_id = old_class_id
            else:
                # New class already exists, map old to existing new
                new_class_id = new_class_name_to_id[new_class_name]
                # Remove old class from list if not used by other mappings
                if old_class_name in new_classes:
                    new_classes.remove(old_class_name)
                del new_class_name_to_id[old_class_name]

            old_to_new_class_id[old_class_id] = new_class_id

        # Add identity mappings for classes not being remapped
        for class_name, old_id in class_name_to_id.items():
            if old_id not in old_to_new_class_id:
                # Class is not being remapped, keep same ID
                old_to_new_class_id[old_id] = new_class_name_to_id[class_name]

        # Apply the mapping with validation
        mapped = df_copy["class_id"].map(old_to_new_class_id)

        # Check if any mappings failed (excluding NaN which are background images)
        non_null_mask = df_copy["class_id"].notna()
        failed_mapping = non_null_mask & mapped.isna()

        if failed_mapping.any():
            unmapped_ids = df_copy.loc[failed_mapping, "class_id"].unique()
            raise ClassMappingError(
                f"Failed to map class IDs: {unmapped_ids}. "
                f"This should not happen - please report this bug."
            )

        df_copy["class_id"] = mapped
        # Convert to int where not null
        df_copy.loc[non_null_mask, "class_id"] = df_copy.loc[
            non_null_mask, "class_id"
        ].astype(int)

        return df_copy, new_classes, new_class_name_to_id

    @staticmethod
    def delete_classes(
        df: pd.DataFrame,
        classes: list[str],
        class_name_to_id: dict[str, int],
        class_names_to_delete: list[str],
    ) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
        """Delete specified classes from the dataset.

        Args:
            df: DataFrame containing the dataset
            classes: Current list of class names
            class_name_to_id: Current class name to ID mapping
            class_names_to_delete: List of class names to delete

        Returns:
            Tuple of (new_df, new_classes, new_class_name_to_id)

        Raises:
            ClassNotFoundError: If a class to delete doesn't exist
        """
        df_copy = df.copy()
        new_classes = classes.copy()

        # Track original class names for mapping later
        original_classes = classes.copy()

        # First, remove the classes and their data
        for class_name in tqdm(class_names_to_delete, desc="Deleting classes"):
            if class_name not in class_name_to_id:
                raise ClassNotFoundError(class_name, classes)

            class_id = class_name_to_id[class_name]
            df_copy = df_copy[df_copy["class_id"] != class_id]
            new_classes.remove(class_name)

        # Rebuild the class_name_to_id mapping with continuous IDs
        new_class_name_to_id = {name: idx for idx, name in enumerate(new_classes)}

        # Update all class_ids in the dataframe using vectorized operation
        # Build mapping from old class_id to new class_id
        old_to_new_class_id = {}
        for old_id, old_class_name in enumerate(original_classes):
            if old_class_name in new_class_name_to_id:
                old_to_new_class_id[old_id] = new_class_name_to_id[old_class_name]

        # Apply the mapping (much faster than iterrows)
        df_copy["class_id"] = df_copy["class_id"].map(old_to_new_class_id).astype(int)

        return df_copy, new_classes, new_class_name_to_id

    @staticmethod
    def create_balanced_dataset(
        df: pd.DataFrame,
        classes: list[str],
        min_samples: int | None = None,
        config: DatasetConfig | None = None,
    ) -> pd.DataFrame:
        """Create a balanced dataset with equal samples per class.

        Args:
            df: DataFrame containing the dataset
            classes: List of class names
            min_samples: Minimum samples per class. If None, uses the minimum count
            config: Dataset configuration for train/valid split

        Returns:
            New balanced DataFrame with train/valid splits

        Raises:
            EmptyDatasetError: If the dataset is empty
        """
        if len(df) == 0:
            raise EmptyDatasetError("Cannot balance an empty dataset")

        if config is None:
            config = DatasetConfig()

        class_counts = df["class_id"].value_counts()
        if len(class_counts) == 0:
            raise EmptyDatasetError("No valid class_ids found in dataset")

        actual_min_samples = (
            min_samples if min_samples is not None else int(class_counts.min())
        )

        # Create a new dataframe with the balanced dataset
        data = []
        rng = config.get_rng()

        for class_id, class_name in tqdm(
            enumerate(classes), total=len(classes), desc="Balancing classes"
        ):
            class_df = df[df["class_id"] == class_id]

            if len(class_df) > actual_min_samples:
                # Use random generator with proper seeding
                indices = rng.choice(
                    len(class_df), size=actual_min_samples, replace=False
                )
                class_df = class_df.iloc[indices].reset_index(drop=True)

            data.extend(class_df.to_dict(orient="records"))

        # Create balanced dataset DataFrame
        balanced_df = pd.DataFrame(data)

        # Extract image_name from image_path
        balanced_df["image_name"] = balanced_df["image_path"].apply(
            lambda x: Path(x).name
        )

        # Get unique image names
        unique_images = balanced_df["image_name"].unique()

        # Create a mapping of image_name to dataset (train or valid) using proper RNG
        image_to_dataset = {
            img: rng.choice(
                ["train", "valid"], p=[config.train_split, config.valid_split]
            )
            for img in unique_images
        }

        # Assign dataset based on image_name
        balanced_df["dataset"] = balanced_df["image_name"].map(image_to_dataset)

        # Drop temporary column
        balanced_df = balanced_df.drop(columns=["image_name"])

        return balanced_df

    @staticmethod
    def add_background_images(
        df: pd.DataFrame,
        background_images_folder: str | Path,
        pct_background: float = 0.2,
        config: DatasetConfig | None = None,
    ) -> pd.DataFrame:
        """Add background (negative) images to the dataset.

        Args:
            df: DataFrame containing the dataset
            background_images_folder: Path to folder containing background images
            pct_background: Percentage of background images relative to min class count
            config: Dataset configuration

        Returns:
            New DataFrame with background images added

        Raises:
            EmptyDatasetError: If the dataset is empty
        """
        if len(df) == 0:
            raise EmptyDatasetError("Cannot add background images to an empty dataset")

        if config is None:
            config = DatasetConfig()

        df_copy = df.copy()
        background_images_folder = Path(background_images_folder)
        background_images = get_image_paths(background_images_folder)

        rng = config.get_rng()
        rng.shuffle(background_images)

        # Calculate number of background images to add
        class_counts = df_copy["class_id"].value_counts()
        num_samples = int(class_counts.min() * pct_background)

        # Collect all background image rows first, then concat once
        new_rows = []
        for image_path in background_images[:num_samples]:
            dataset = rng.choice(
                ["train", "valid"], p=[config.train_split, config.valid_split]
            )
            new_rows.append(
                {
                    "dataset": dataset,
                    "image_path": image_path,
                    "xywh": None,
                    "class_id": None,
                }
            )

        # Single concat operation (much faster than loop)
        if new_rows:
            df_copy = pd.concat([df_copy, pd.DataFrame(new_rows)], ignore_index=True)

        return df_copy
