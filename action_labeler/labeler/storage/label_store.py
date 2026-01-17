"""Core data storage for labeled detections.

This module provides efficient storage and retrieval of labeled detections
with proper indexing and deduplication.
"""

from typing import Any

import pandas as pd

from action_labeler.labeler.storage.metadata import LabeledDetection, LabelMetadata


class LabelStore:
    """Efficient storage for labeled detections with metadata.

    Uses pandas DataFrame for efficient operations with proper indexing
    for fast lookups and deduplication.

    The store maintains:
    - Core detection data (image_path, xywh, segmentation, label)
    - Rich metadata for each label
    - Indexes for fast deduplication checks
    """

    def __init__(self) -> None:
        """Initialize empty label store."""
        # Main DataFrame with detection data
        self.df = pd.DataFrame(
            columns=["image_path", "xywh", "segmentation_points", "label", "metadata"]
        )

        # Index for fast deduplication: (image_path, xywh_str) -> row index
        self._detection_index: dict[tuple[str, str], int] = {}

        # Batch buffer for efficient additions
        self._batch_buffer: list[dict[str, Any]] = []
        self._batch_size = 1000  # Flush every N additions

    def add(self, detection: LabeledDetection) -> bool:
        """Add a labeled detection to the store.

        Args:
            detection: Labeled detection to add

        Returns:
            True if added, False if duplicate (already exists)
        """
        # Check for duplicates using index
        detection_key = self._get_detection_key(detection.image_path, detection.xywh)

        if detection_key in self._detection_index:
            return False  # Duplicate

        # Add to batch buffer
        self._batch_buffer.append(detection.to_dict())

        # Update index (will point to future row after flush)
        future_index = len(self.df) + len(self._batch_buffer) - 1
        self._detection_index[detection_key] = future_index

        # Flush if buffer is full
        if len(self._batch_buffer) >= self._batch_size:
            self.flush()

        return True

    def add_batch(self, detections: list[LabeledDetection]) -> int:
        """Add multiple detections efficiently.

        Args:
            detections: List of labeled detections

        Returns:
            Number of detections actually added (excluding duplicates)
        """
        added_count = 0

        for detection in detections:
            if self.add(detection):
                added_count += 1

        return added_count

    def flush(self) -> None:
        """Flush batch buffer to DataFrame.

        This should be called before querying to ensure all data is available.
        """
        if not self._batch_buffer:
            return

        # Convert buffer to DataFrame
        new_df = pd.DataFrame(self._batch_buffer)

        # Append to main DataFrame
        self.df = pd.concat([self.df, new_df], ignore_index=True)

        # Clear buffer
        self._batch_buffer = []

    def exists(self, image_path: str, xywh: list[float]) -> bool:
        """Check if a detection already has a label.

        Args:
            image_path: Path to image
            xywh: Bounding box coordinates

        Returns:
            True if detection exists in store
        """
        detection_key = self._get_detection_key(image_path, xywh)
        return detection_key in self._detection_index

    def get(self, image_path: str, xywh: list[float]) -> LabeledDetection | None:
        """Retrieve a labeled detection.

        Args:
            image_path: Path to image
            xywh: Bounding box coordinates

        Returns:
            LabeledDetection if found, None otherwise
        """
        # Ensure buffer is flushed
        self.flush()

        detection_key = self._get_detection_key(image_path, xywh)

        if detection_key not in self._detection_index:
            return None

        row_idx = self._detection_index[detection_key]
        row = self.df.iloc[row_idx]

        return self._row_to_labeled_detection(row)

    def get_all(self) -> list[LabeledDetection]:
        """Get all labeled detections.

        Returns:
            List of all LabeledDetection objects
        """
        self.flush()

        detections = []
        for _, row in self.df.iterrows():
            detections.append(self._row_to_labeled_detection(row))

        return detections

    def filter_by_experiment(self, experiment_id: str) -> list[LabeledDetection]:
        """Get all detections from a specific experiment.

        Args:
            experiment_id: Experiment ID to filter by

        Returns:
            List of LabeledDetection from that experiment
        """
        self.flush()

        detections = []
        for _, row in self.df.iterrows():
            metadata = LabelMetadata.from_dict(row["metadata"])
            if metadata.experiment_id == experiment_id:
                detections.append(self._row_to_labeled_detection(row))

        return detections

    def filter_by_label(self, label: str) -> list[LabeledDetection]:
        """Get all detections with a specific label.

        Args:
            label: Label to filter by

        Returns:
            List of LabeledDetection with that label
        """
        self.flush()

        filtered_df = self.df[self.df["label"] == label]

        detections = []
        for _, row in filtered_df.iterrows():
            detections.append(self._row_to_labeled_detection(row))

        return detections

    def filter_by_image(self, image_path: str) -> list[LabeledDetection]:
        """Get all detections from a specific image.

        Args:
            image_path: Image path to filter by

        Returns:
            List of LabeledDetection from that image
        """
        self.flush()

        filtered_df = self.df[self.df["image_path"] == image_path]

        detections = []
        for _, row in filtered_df.iterrows():
            detections.append(self._row_to_labeled_detection(row))

        return detections

    def filter_invalid(self) -> list[LabeledDetection]:
        """Get all detections that failed validation.

        Returns:
            List of invalid LabeledDetection
        """
        self.flush()

        invalid_detections = []
        for _, row in self.df.iterrows():
            metadata = LabelMetadata.from_dict(row["metadata"])
            if not metadata.is_valid:
                invalid_detections.append(self._row_to_labeled_detection(row))

        return invalid_detections

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the label store.

        Returns:
            Dictionary with statistics
        """
        self.flush()

        if len(self.df) == 0:
            return {
                "total_labels": 0,
                "unique_images": 0,
                "unique_labels": 0,
                "label_distribution": {},
                "experiment_distribution": {},
                "invalid_count": 0,
            }

        # Extract metadata for analysis
        metadata_list = [LabelMetadata.from_dict(m) for m in self.df["metadata"]]

        # Count invalid labels
        invalid_count = sum(1 for m in metadata_list if not m.is_valid)

        # Experiment distribution
        experiment_counts = {}
        for m in metadata_list:
            experiment_counts[m.experiment_id] = (
                experiment_counts.get(m.experiment_id, 0) + 1
            )

        return {
            "total_labels": len(self.df),
            "unique_images": self.df["image_path"].nunique(),
            "unique_labels": self.df["label"].nunique(),
            "label_distribution": self.df["label"].value_counts().to_dict(),
            "experiment_distribution": experiment_counts,
            "invalid_count": invalid_count,
        }

    def clear(self) -> None:
        """Clear all data from the store."""
        self.df = pd.DataFrame(
            columns=["image_path", "xywh", "segmentation_points", "label", "metadata"]
        )
        self._detection_index = {}
        self._batch_buffer = []

    def __len__(self) -> int:
        """Get number of labeled detections.

        Returns:
            Total count including buffer
        """
        return len(self.df) + len(self._batch_buffer)

    def _get_detection_key(self, image_path: str, xywh: list[float]) -> tuple[str, str]:
        """Create a hashable key for a detection.

        Args:
            image_path: Image path
            xywh: Bounding box coordinates

        Returns:
            Tuple of (image_path, xywh_string)
        """
        # Convert xywh to string for hashing
        xywh_str = " ".join(map(str, xywh))
        return (image_path, xywh_str)

    def _row_to_labeled_detection(self, row: pd.Series) -> LabeledDetection:
        """Convert DataFrame row to LabeledDetection.

        Args:
            row: DataFrame row

        Returns:
            LabeledDetection object
        """
        return LabeledDetection(
            image_path=row["image_path"],
            xywh=row["xywh"],
            segmentation_points=row["segmentation_points"],
            label=row["label"],
            metadata=LabelMetadata.from_dict(row["metadata"]),
        )

    def _rebuild_index(self) -> None:
        """Rebuild the detection index from DataFrame.

        This should be called after loading data from disk.
        """
        self._detection_index = {}

        for idx, row in self.df.iterrows():
            detection_key = self._get_detection_key(row["image_path"], row["xywh"])
            self._detection_index[detection_key] = idx

    def to_simple_dataframe(self) -> pd.DataFrame:
        """Convert to a simple DataFrame without nested metadata.

        Useful for export and analysis that doesn't need full metadata.

        Returns:
            DataFrame with flattened structure
        """
        self.flush()

        if len(self.df) == 0:
            return pd.DataFrame(
                columns=["image_path", "xywh", "segmentation_points", "label"]
            )

        # Create simple DataFrame
        simple_df = self.df[
            ["image_path", "xywh", "segmentation_points", "label"]
        ].copy()

        return simple_df

    def merge(self, other: "LabelStore") -> int:
        """Merge another label store into this one.

        Only adds detections that don't already exist (deduplication).

        Args:
            other: Another LabelStore to merge

        Returns:
            Number of new detections added
        """
        other.flush()
        added_count = 0

        for _, row in other.df.iterrows():
            detection = LabeledDetection(
                image_path=row["image_path"],
                xywh=row["xywh"],
                segmentation_points=row["segmentation_points"],
                label=row["label"],
                metadata=LabelMetadata.from_dict(row["metadata"]),
            )

            if self.add(detection):
                added_count += 1

        return added_count
