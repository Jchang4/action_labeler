from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .columns import DatasetColumns
from .filter import DatasetFilterMixin
from .plot import DatasetPlotMixin

if TYPE_CHECKING:
    from ..labeler import LabelResult


class Dataset(DatasetPlotMixin, DatasetFilterMixin):
    def __init__(self, df: pd.DataFrame):
        self._validate(df)
        self.df = df

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Assert all required columns exist."""
        missing = DatasetColumns.REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    @classmethod
    def from_label_results(cls, results: list[LabelResult]) -> Dataset:
        """Build Dataset from ActionLabeler output.

        Groups by image_path and assigns detection_index (0, 1, 2...) per image.
        """
        rows = []
        for result in results:
            rows.append(
                {
                    DatasetColumns.IMAGE_PATH: result.image_path,
                    DatasetColumns.DETECTION: result.detection,
                    DatasetColumns.RESPONSE: result.response,
                }
            )

        if not rows:
            df = pd.DataFrame(
                columns=[
                    DatasetColumns.IMAGE_PATH,
                    DatasetColumns.DETECTION_INDEX,
                    DatasetColumns.DETECTION,
                    DatasetColumns.RESPONSE,
                ]
            )
        else:
            df = pd.DataFrame(rows)
            df[DatasetColumns.DETECTION_INDEX] = df.groupby(
                DatasetColumns.IMAGE_PATH
            ).cumcount()
        return cls(df)

    def save(self, path: Path) -> None:
        """Pickle the DataFrame to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.df, f)

    @classmethod
    def load(cls, path: Path) -> Dataset:
        """Load from pickled DataFrame. Runs _validate on load."""
        with open(path, "rb") as f:
            df = pickle.load(f)  # noqa: S301
        return cls(df)

    def response_field(self, field_name: str) -> pd.Series:
        """Extract a field from all response objects as a Series."""
        return self.df[DatasetColumns.RESPONSE].apply(
            lambda r: getattr(r, field_name)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"Dataset({len(self.df)} rows)"
