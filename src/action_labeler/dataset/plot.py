from __future__ import annotations

import math

from matplotlib import colormaps as mpl_colormaps
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

        # Build a consistent color map: each action gets a unique color
        unique_actions = sorted(self.df[DatasetColumns.ACTION].unique())
        cmap = mpl_colormaps["tab10"]
        color_map = {
            act: tuple(int(c * 255) for c in cmap(i % cmap.N)[:3])
            for i, act in enumerate(unique_actions)
        }

        # Group all rows by image so we can draw every detection per image
        all_by_image = self.df.groupby(DatasetColumns.IMAGE_PATH)

        cols = math.ceil(math.sqrt(len(sample)))
        rows = math.ceil(len(sample) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, (_, row) in zip(axes, sample.iterrows()):
            image_path = row[DatasetColumns.IMAGE_PATH]
            action_label = row[DatasetColumns.ACTION]

            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Draw ALL detections for this image
            for _, sibling in all_by_image.get_group(image_path).iterrows():
                det = sibling[DatasetColumns.DETECTION]
                act = sibling[DatasetColumns.ACTION]
                color = color_map.get(act, (255, 0, 0))
                draw.rectangle(det.xyxy, outline=color, width=2)
                draw.text(
                    (det.x1, max(0, det.y1 - 12)), act, fill=color
                )

            ax.imshow(img)
            ax.set_title(action_label, fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for ax in axes[len(sample):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def detection_stats(self) -> pd.DataFrame:
        """Return average detection size statistics grouped by action.

        Returns a DataFrame with columns: avg_width, avg_height, avg_area, count.
        Area is width * height (fraction of image area in normalized space).
        Sorted by avg_area descending.
        """
        col = DatasetColumns

        stats = self.df.groupby(col.ACTION)[col.DETECTION].agg(
            avg_width=lambda dets: dets.apply(lambda d: d.width).mean(),
            avg_height=lambda dets: dets.apply(lambda d: d.height).mean(),
            avg_area=lambda dets: dets.apply(lambda d: d.width * d.height).mean(),
            count="count",
        )
        return stats.sort_values("avg_area", ascending=False)

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
