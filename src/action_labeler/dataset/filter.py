from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .dataset import Dataset

from .columns import DatasetColumns


class DatasetFilterMixin:
    """Filtering methods for Dataset. All methods mutate in-place and reset the index."""

    df: pd.DataFrame

    def remove_class(self, class_name: str) -> None:
        """Remove rows where action == class_name."""
        col = DatasetColumns.ACTION
        self.df = self.df[self.df[col] != class_name].reset_index(drop=True)

    def keep_classes(self, class_names: list[str]) -> None:
        """Keep only rows where action is in class_names."""
        col = DatasetColumns.ACTION
        self.df = self.df[self.df[col].isin(class_names)].reset_index(drop=True)

    def remove_image(self, image_path: Path) -> None:
        """Remove all rows for a given image path."""
        col = DatasetColumns.IMAGE_PATH
        self.df = self.df[self.df[col] != image_path].reset_index(drop=True)
