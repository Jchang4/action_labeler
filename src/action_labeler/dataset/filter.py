from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .dataset import Dataset

from .columns import DatasetColumns


class DatasetFilterMixin:
    """Filtering methods for Dataset. All methods mutate in-place and reset the index."""

    df: pd.DataFrame

    def balance(
        self,
        upsample: float | dict[str, float] = 1.0,
        seed: int | None = None,
    ) -> None:
        """Downsample overrepresented classes by dropping entire images.

        Args:
            upsample: Multiplier applied to min_count to compute per-class targets.
                Float applies uniformly; dict allows per-class multipliers.
            seed: Random seed for reproducible image sampling.
        """
        if len(self.df) == 0:
            return

        col_action = DatasetColumns.ACTION
        col_image = DatasetColumns.IMAGE_PATH

        # Count detections per action
        action_counts = self.df[col_action].value_counts().to_dict()
        min_count = min(action_counts.values())

        # Compute per-class targets
        targets: dict[str, int] = {}
        for action, actual in action_counts.items():
            if isinstance(upsample, dict):
                mult = upsample.get(action, 1.0)
            else:
                mult = upsample
            targets[action] = min(math.ceil(min_count * mult), actual)

        # Process actions from rarest to most common
        sorted_actions = sorted(action_counts, key=lambda a: (action_counts[a], a))

        rng = random.Random(seed)
        kept_images: set[Path] = set()

        for action in sorted_actions:
            target = targets[action]

            # Count detections of this action already in the keep set
            if kept_images:
                kept_mask = self.df[col_image].isin(kept_images)
                already_kept = int((self.df.loc[kept_mask, col_action] == action).sum())
            else:
                already_kept = 0

            if already_kept >= target:
                continue

            # Candidate images: have this action and aren't already kept
            action_mask = self.df[col_action] == action
            candidate_images = list(
                set(self.df.loc[action_mask, col_image]) - kept_images
            )
            rng.shuffle(candidate_images)

            current = already_kept
            for img in candidate_images:
                if current >= target:
                    break
                kept_images.add(img)
                # Count how many detections of this action this image contributes
                img_action_count = int(
                    ((self.df[col_image] == img) & action_mask).sum()
                )
                current += img_action_count

        self.df = self.df[self.df[col_image].isin(kept_images)].reset_index(drop=True)

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
