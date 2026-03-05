from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..types import Detection, LabelResult
from .columns import DatasetColumns
from .filter import DatasetFilterMixin
from .plot import DatasetPlotMixin


class Dataset(DatasetPlotMixin, DatasetFilterMixin):
    def __init__(self, df: pd.DataFrame | None = None):
        if df is None:
            df = pd.DataFrame(columns=[c for c in DatasetColumns.REQUIRED])
        self._validate(df)
        self.df = df

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Assert all required columns exist."""
        missing = DatasetColumns.REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def add_rows(
        self,
        image_path: Path,
        detections: list[Detection],
        results: list[LabelResult],
    ) -> None:
        """Append rows for all detections in an image.

        Deduplicates by (image_path, detection) keeping the latest result.
        Detection indices are recomputed per image after dedup.
        """
        rows = pd.DataFrame(
            [
                {
                    DatasetColumns.IMAGE_PATH: image_path,
                    DatasetColumns.DETECTION_INDEX: 0,  # recomputed below
                    DatasetColumns.DETECTION: det,
                    DatasetColumns.ACTION: result.action,
                    DatasetColumns.RESPONSE: result.response,
                }
                for det, result in zip(detections, results)
            ]
        )
        self.df = pd.concat([self.df, rows], ignore_index=True)

        # Drop older duplicates, keep latest
        dupes = self.df.duplicated(
            subset=[DatasetColumns.IMAGE_PATH, DatasetColumns.DETECTION],
            keep="last",
        )
        if dupes.any():
            self.df = self.df[~dupes].reset_index(drop=True)

        # Recompute detection_index for this image
        mask = self.df[DatasetColumns.IMAGE_PATH] == image_path
        self.df.loc[mask, DatasetColumns.DETECTION_INDEX] = range(mask.sum())

    def has_row(self, image_path: Path, detection: Detection) -> bool:
        """Check if a row with this (image_path, detection) pair exists."""
        mask = (self.df[DatasetColumns.IMAGE_PATH] == image_path) & (
            self.df[DatasetColumns.DETECTION] == detection
        )
        return bool(mask.any())

    def save(self, path: Path) -> None:
        """Pickle the DataFrame to disk."""
        self.df.to_pickle(path)

    @classmethod
    def load(cls, path: Path) -> Dataset:
        """Load from pickled DataFrame. Runs _validate on load."""
        return cls(pd.read_pickle(path))

    def response_field(self, field_name: str) -> pd.Series:
        """Extract a field from all response objects as a Series."""
        return self.df[DatasetColumns.RESPONSE].apply(lambda r: getattr(r, field_name))

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"Dataset({len(self.df)} rows)"
