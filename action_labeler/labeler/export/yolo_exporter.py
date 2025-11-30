"""YOLO v8 dataset exporter.

Exports labeled detections to YOLO v8 format with train/valid split.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from action_labeler.datasets.dataset_config import DatasetConfig
from action_labeler.labeler.export.base import IDatasetExporter
from action_labeler.labeler.storage.label_store import LabelStore


class YoloV8Exporter(IDatasetExporter):
    """Exporter for YOLO v8 dataset format.

    Creates directory structure:
    ```
    output_folder/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── valid/
        ├── images/
        └── labels/
    ```
    """

    def __init__(self, config: DatasetConfig | None = None):
        """Initialize YOLO exporter.

        Args:
            config: Dataset configuration (train/valid split, random seed)
        """
        self.config = config or DatasetConfig()

    def export(
        self,
        label_store: LabelStore,
        output_path: str | Path,
        delete_existing: bool = False,
    ) -> None:
        """Export label store to YOLO v8 format.

        Args:
            label_store: Store containing labeled detections
            output_path: Path to output directory
            delete_existing: Whether to delete existing output directory

        Raises:
            ValueError: If validation fails
        """
        output_path = Path(output_path)

        # Validate before export
        is_valid, errors = self.validate_export(label_store)
        if not is_valid:
            raise ValueError(
                f"Cannot export to YOLO format:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Create directory structure
        self._create_directory_structure(output_path, delete_existing)

        # Get simple DataFrame for export
        label_store.flush()
        df = label_store.to_simple_dataframe()

        # Get unique labels and create class mapping
        unique_labels = sorted(df["label"].unique())
        class_name_to_id = {name: idx for idx, name in enumerate(unique_labels)}

        # Assign train/valid splits
        df = self._assign_splits(df)

        # Export images and labels
        self._export_data(df, output_path, class_name_to_id)

        # Create data.yaml
        self._create_data_yaml(output_path, unique_labels)

    def get_format_name(self) -> str:
        """Get the name of this export format."""
        return "yolov8"

    def validate_export(self, label_store: LabelStore) -> tuple[bool, list[str]]:
        """Validate that label store can be exported to YOLO format.

        Args:
            label_store: Store to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if store is empty
        if len(label_store) == 0:
            errors.append("Label store is empty")
            return False, errors

        # Check if all detections have valid labels
        label_store.flush()
        if label_store.df["label"].isna().any():
            errors.append("Some detections have missing labels")

        # Check for empty labels
        empty_labels = (label_store.df["label"] == "").sum()
        if empty_labels > 0:
            errors.append(f"{empty_labels} detections have empty labels")

        # Check if xywh coordinates are valid
        for idx, row in label_store.df.iterrows():
            xywh = row["xywh"]
            if not isinstance(xywh, (list, tuple)) or len(xywh) != 4:
                errors.append(f"Invalid xywh format at index {idx}: {xywh}")
                break  # Don't spam with too many errors

            # Check if coordinates are in valid range
            if any(x < 0 or x > 1 for x in xywh):
                errors.append(
                    f"xywh coordinates out of range [0, 1] at index {idx}: {xywh}"
                )
                break

        is_valid = len(errors) == 0
        return is_valid, errors

    def _create_directory_structure(
        self, output_path: Path, delete_existing: bool
    ) -> None:
        """Create YOLO directory structure.

        Args:
            output_path: Base output directory
            delete_existing: Whether to delete existing directory
        """
        if output_path.exists() and delete_existing:
            shutil.rmtree(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for split in ["train", "valid"]:
            (output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    def _assign_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign train/valid splits to images.

        Args:
            df: DataFrame with labeled detections

        Returns:
            DataFrame with 'dataset' column added
        """
        df = df.copy()

        # Get unique images
        unique_images = df["image_path"].unique()

        # Create split assignment using configured RNG
        rng = self.config.get_rng()
        image_to_split = {}

        for image_path in unique_images:
            split = rng.choice(
                ["train", "valid"],
                p=[self.config.train_split, self.config.valid_split],
            )
            image_to_split[image_path] = split

        # Assign splits
        df["dataset"] = df["image_path"].map(image_to_split)

        return df

    def _export_data(
        self,
        df: pd.DataFrame,
        output_path: Path,
        class_name_to_id: dict[str, int],
    ) -> None:
        """Export images and labels to YOLO format.

        Args:
            df: DataFrame with labeled detections and splits
            output_path: Base output directory
            class_name_to_id: Mapping of class names to IDs
        """
        # Group by image
        grouped = df.groupby("image_path")

        for image_path, group in tqdm(grouped, desc="Exporting images"):
            image_path = Path(image_path)

            # Get split for this image
            split = group.iloc[0]["dataset"]

            # Copy image
            dest_image_dir = output_path / split / "images"
            dest_image_path = dest_image_dir / image_path.name

            if image_path.exists():
                shutil.copy2(image_path, dest_image_path)
            else:
                print(f"Warning: Image not found: {image_path}")
                continue

            # Create label file
            dest_label_dir = output_path / split / "labels"
            dest_label_path = dest_label_dir / f"{image_path.stem}.txt"

            self._write_label_file(dest_label_path, group, class_name_to_id)

    def _write_label_file(
        self,
        label_path: Path,
        detections: pd.DataFrame,
        class_name_to_id: dict[str, int],
    ) -> None:
        """Write YOLO format label file.

        Args:
            label_path: Path to label file
            detections: DataFrame with detections for this image
            class_name_to_id: Mapping of class names to IDs
        """
        lines = []

        for _, row in detections.iterrows():
            label = row["label"]
            xywh = row["xywh"]

            # Get class ID
            class_id = class_name_to_id.get(label)
            if class_id is None:
                print(f"Warning: Unknown label '{label}', skipping")
                continue

            # Format: class_id x_center y_center width height
            line = f"{class_id} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}"

            # Add segmentation points if available
            seg_points = row.get("segmentation_points", [])
            if seg_points and len(seg_points) > 0:
                # Flatten points: [x1, y1, x2, y2, ...]
                flat_points = []
                for point in seg_points:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        flat_points.extend([point[0], point[1]])

                if flat_points:
                    points_str = " ".join(map(str, flat_points))
                    line += f" {points_str}"

            lines.append(line)

        # Write to file
        label_path.write_text("\n".join(lines))

    def _create_data_yaml(self, output_path: Path, classes: list[str]) -> None:
        """Create data.yaml configuration file.

        Args:
            output_path: Base output directory
            classes: List of class names
        """
        data_yaml = {
            "path": str(output_path.absolute()),
            "train": "train/images",
            "val": "valid/images",
            "names": classes,
        }

        yaml_path = output_path / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)


class YoloV8BalancedExporter(YoloV8Exporter):
    """YOLO exporter with class balancing.

    Ensures equal number of samples per class before export.
    """

    def __init__(
        self,
        config: DatasetConfig | None = None,
        min_samples: int | None = None,
    ):
        """Initialize balanced YOLO exporter.

        Args:
            config: Dataset configuration
            min_samples: Minimum samples per class (None = use minimum class count)
        """
        super().__init__(config)
        self.min_samples = min_samples

    def export(
        self,
        label_store: LabelStore,
        output_path: str | Path,
        delete_existing: bool = False,
    ) -> None:
        """Export balanced dataset to YOLO v8 format.

        Args:
            label_store: Store containing labeled detections
            output_path: Path to output directory
            delete_existing: Whether to delete existing output directory
        """
        # Balance the dataset first
        balanced_store = self._balance_dataset(label_store)

        # Export using parent method
        super().export(balanced_store, output_path, delete_existing)

    def _balance_dataset(self, label_store: LabelStore) -> LabelStore:
        """Create balanced dataset with equal samples per class.

        Args:
            label_store: Original label store

        Returns:
            New balanced label store
        """
        label_store.flush()
        df = label_store.to_simple_dataframe()

        # Get class counts
        class_counts = df["label"].value_counts()

        # Determine target count
        target_count = self.min_samples if self.min_samples else int(class_counts.min())

        # Sample from each class
        balanced_rows = []
        rng = self.config.get_rng()

        for label in class_counts.index:
            label_df = df[df["label"] == label]

            if len(label_df) > target_count:
                # Downsample
                sampled_df = label_df.sample(
                    n=target_count, random_state=rng.integers(0, 2**31)
                )
            else:
                # Keep all (could also upsample if desired)
                sampled_df = label_df

            balanced_rows.append(sampled_df)

        # Combine
        balanced_df = pd.concat(balanced_rows, ignore_index=True)

        # Create new label store
        # We need to reconstruct LabeledDetection objects
        # For simplicity, create a new store with minimal metadata
        from action_labeler.labeler.storage.metadata import (
            LabeledDetection,
            LabelMetadata,
        )

        balanced_store = LabelStore()

        for _, row in balanced_df.iterrows():
            # Create minimal metadata
            metadata = LabelMetadata(
                experiment_id="balanced_export",
                model_name="unknown",
                prompt_version="1.0",
            )

            detection = LabeledDetection(
                image_path=row["image_path"],
                xywh=row["xywh"],
                segmentation_points=row.get("segmentation_points", []),
                label=row["label"],
                metadata=metadata,
            )

            balanced_store.add(detection)

        balanced_store.flush()
        return balanced_store
