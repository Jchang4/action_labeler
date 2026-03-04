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

    def remove_class(self, class_name: str, field: str = "action") -> None:
        """Remove rows where response.{field} == class_name."""
        col = DatasetColumns.RESPONSE
        mask = self.df[col].apply(lambda r: getattr(r, field, None) != class_name)
        self.df = self.df[mask].reset_index(drop=True)

    def keep_classes(self, class_names: list[str], field: str = "action") -> None:
        """Keep only rows where response.{field} is in class_names."""
        col = DatasetColumns.RESPONSE
        mask = self.df[col].apply(
            lambda r: getattr(r, field, None) in class_names
        )
        self.df = self.df[mask].reset_index(drop=True)

    def remove_image(self, image_path: Path) -> None:
        """Remove all rows for a given image path."""
        col = DatasetColumns.IMAGE_PATH
        self.df = self.df[self.df[col] != image_path].reset_index(drop=True)
