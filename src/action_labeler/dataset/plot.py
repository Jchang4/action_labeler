from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

from .columns import DatasetColumns


class DatasetPlotMixin:
    """Plotting methods for Dataset. Read-only — never mutates self.df."""

    df: pd.DataFrame

    def plot_grid(
        self,
        n: int = 16,
        action: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Display a grid of sample images with bounding boxes and action labels.

        Args:
            n: Number of images to display.
            action: If set, only show rows with this action.
            seed: Random seed for reproducible sampling.
        """
        df = self.df
        if action is not None:
            df = df[df[DatasetColumns.ACTION] == action]

        if len(df) == 0:
            print("No rows to plot.")
            return

        sample = df.sample(n=min(n, len(df)), random_state=seed)

        cols = math.ceil(math.sqrt(len(sample)))
        rows = math.ceil(len(sample) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, (_, row) in zip(axes, sample.iterrows()):
            image_path = row[DatasetColumns.IMAGE_PATH]
            detection = row[DatasetColumns.DETECTION]
            action_label = row[DatasetColumns.ACTION]

            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle(detection.xyxy, outline="red", width=2)
            draw.text((detection.x1, max(0, detection.y1 - 12)), action_label, fill="red")

            ax.imshow(img)
            ax.set_title(action_label, fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for ax in axes[len(sample):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_distribution(self) -> None:
        """Bar chart of action class counts."""
        if len(self.df) == 0:
            print("No rows to plot.")
            return

        counts = self.df[DatasetColumns.ACTION].value_counts()
        ax = counts.plot.bar()
        ax.set_xlabel("Action")
        ax.set_ylabel("Count")
        ax.set_title("Action Distribution")
        plt.tight_layout()
        plt.show()
