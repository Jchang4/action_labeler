"""Persistence layer for saving and loading labeled data.

Supports multiple formats:
- Pickle (for full metadata preservation)
- Parquet (efficient, preserves types)
- JSON (human-readable, git-friendly)
"""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from action_labeler.labeler.storage.label_store import LabelStore
from action_labeler.labeler.storage.metadata import LabeledDetection


class LabelPersistence:
    """Handles saving and loading of label stores to/from disk.

    Supports multiple formats with different trade-offs:
    - Pickle: Fast, preserves Python objects, not human-readable
    - Parquet: Efficient, preserves types, widely supported
    - JSON: Human-readable, git-friendly, larger file size
    """

    @staticmethod
    def save_pickle(store: LabelStore, path: str | Path) -> None:
        """Save label store to pickle format.

        Args:
            store: LabelStore to save
            path: Path to save file
        """
        store.flush()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save DataFrame directly (includes all metadata)
        with open(path, "wb") as f:
            pickle.dump(store.df, f)

    @staticmethod
    def load_pickle(path: str | Path) -> LabelStore:
        """Load label store from pickle format.

        Args:
            path: Path to pickle file

        Returns:
            Loaded LabelStore
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")

        with open(path, "rb") as f:
            df = pickle.load(f)

        store = LabelStore()
        store.df = df
        store._rebuild_index()

        return store

    @staticmethod
    def save_parquet(store: LabelStore, path: str | Path) -> None:
        """Save label store to parquet format.

        Parquet doesn't support nested objects, so metadata is serialized to JSON.

        Args:
            store: LabelStore to save
            path: Path to save file
        """
        store.flush()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if len(store.df) == 0:
            # Create empty parquet with correct schema
            empty_df = pd.DataFrame(
                columns=[
                    "image_path",
                    "xywh",
                    "segmentation_points",
                    "label",
                    "metadata_json",
                ]
            )
            empty_df.to_parquet(path, index=False)
            return

        # Convert to parquet-compatible format
        df_copy = store.df.copy()

        # Serialize nested structures to JSON strings
        df_copy["xywh"] = df_copy["xywh"].apply(json.dumps)
        df_copy["segmentation_points"] = df_copy["segmentation_points"].apply(
            json.dumps
        )
        df_copy["metadata_json"] = df_copy["metadata"].apply(json.dumps)
        df_copy = df_copy.drop(columns=["metadata"])

        df_copy.to_parquet(path, index=False)

    @staticmethod
    def load_parquet(path: str | Path) -> LabelStore:
        """Load label store from parquet format.

        Args:
            path: Path to parquet file

        Returns:
            Loaded LabelStore
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path)

        if len(df) == 0:
            return LabelStore()

        # Deserialize JSON strings back to objects
        df["xywh"] = df["xywh"].apply(json.loads)
        df["segmentation_points"] = df["segmentation_points"].apply(json.loads)
        df["metadata"] = df["metadata_json"].apply(json.loads)
        df = df.drop(columns=["metadata_json"])

        store = LabelStore()
        store.df = df
        store._rebuild_index()

        return store

    @staticmethod
    def save_json(store: LabelStore, path: str | Path, indent: int = 2) -> None:
        """Save label store to JSON format.

        Human-readable and git-friendly, but larger file size.

        Args:
            store: LabelStore to save
            path: Path to save file
            indent: JSON indentation (None for compact)
        """
        store.flush()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert all detections to dictionaries
        detections_data = []
        for _, row in store.df.iterrows():
            detection = LabeledDetection(
                image_path=row["image_path"],
                xywh=row["xywh"],
                segmentation_points=row["segmentation_points"],
                label=row["label"],
                metadata=row["metadata"],
            )
            detections_data.append(detection.to_dict())

        # Save as JSON
        with open(path, "w") as f:
            json.dump({"detections": detections_data, "version": "2.0"}, f, indent=indent)

    @staticmethod
    def load_json(path: str | Path) -> LabelStore:
        """Load label store from JSON format.

        Args:
            path: Path to JSON file

        Returns:
            Loaded LabelStore
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        # Support both old and new formats
        if "detections" in data:
            detections_data = data["detections"]
        else:
            # Old format: assume data is list of detections
            detections_data = data

        # Create labeled detections
        detections = [
            LabeledDetection.from_dict(det_data) for det_data in detections_data
        ]

        # Add to store
        store = LabelStore()
        store.add_batch(detections)
        store.flush()

        return store

    @staticmethod
    def save(
        store: LabelStore, path: str | Path, format: str = "auto"
    ) -> None:
        """Save label store with auto-detected or specified format.

        Args:
            store: LabelStore to save
            path: Path to save file
            format: Format to use ("auto", "pickle", "parquet", "json")

        Raises:
            ValueError: If format is invalid
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".pkl" or suffix == ".pickle":
                format = "pickle"
            elif suffix == ".parquet":
                format = "parquet"
            elif suffix == ".json":
                format = "json"
            else:
                # Default to pickle
                format = "pickle"

        if format == "pickle":
            LabelPersistence.save_pickle(store, path)
        elif format == "parquet":
            LabelPersistence.save_parquet(store, path)
        elif format == "json":
            LabelPersistence.save_json(store, path)
        else:
            raise ValueError(
                f"Invalid format: {format}. Must be 'pickle', 'parquet', or 'json'"
            )

    @staticmethod
    def load(path: str | Path, format: str = "auto") -> LabelStore:
        """Load label store with auto-detected or specified format.

        Args:
            path: Path to file
            format: Format to use ("auto", "pickle", "parquet", "json")

        Returns:
            Loaded LabelStore

        Raises:
            ValueError: If format is invalid
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".pkl" or suffix == ".pickle":
                format = "pickle"
            elif suffix == ".parquet":
                format = "parquet"
            elif suffix == ".json":
                format = "json"
            else:
                # Try pickle as default
                format = "pickle"

        if format == "pickle":
            return LabelPersistence.load_pickle(path)
        elif format == "parquet":
            return LabelPersistence.load_parquet(path)
        elif format == "json":
            return LabelPersistence.load_json(path)
        else:
            raise ValueError(
                f"Invalid format: {format}. Must be 'pickle', 'parquet', or 'json'"
            )


def migrate_old_dataset(old_pickle_path: str | Path) -> LabelStore:
    """Migrate old LabelerDataset pickle to new LabelStore.

    Converts old format (without metadata) to new format with minimal metadata.

    Args:
        old_pickle_path: Path to old classification.pickle file

    Returns:
        New LabelStore with migrated data
    """
    from action_labeler.labeler.storage.metadata import LabelMetadata

    path = Path(old_pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"Old pickle file not found: {path}")

    # Load old DataFrame
    with open(path, "rb") as f:
        old_df = pickle.load(f)

    # Create new store
    store = LabelStore()

    # Convert each row to new format
    for _, row in old_df.iterrows():
        # Create minimal metadata for migrated data
        metadata = LabelMetadata(
            experiment_id="migrated",
            model_name="unknown",
            prompt_version="unknown",
            processing_mode="single",
            raw_model_response=row.get("action", ""),
        )

        detection = LabeledDetection(
            image_path=str(row["image_path"]),
            xywh=row["xywh"],
            segmentation_points=row.get("segmentation_points", []),
            label=row.get("action", ""),
            metadata=metadata,
        )

        store.add(detection)

    store.flush()
    return store
