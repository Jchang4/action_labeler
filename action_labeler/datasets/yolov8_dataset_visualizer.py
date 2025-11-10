"""Visualization operations for YoloV8Dataset.

This module handles plotting and visualization of dataset information.
"""

import matplotlib.pyplot as plt
import pandas as pd


class YoloV8DatasetVisualizer:
    """Handles visualization operations for YoloV8Dataset."""

    @staticmethod
    def plot_class_distribution(
        df: pd.DataFrame, classes: list[str], title: str | None = None
    ) -> None:
        """Plot the distribution of classes across train and validation sets.

        Args:
            df: DataFrame containing the dataset
            classes: List of class names
            title: Optional custom title for the plot
        """
        # Create a single figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get class names for all samples
        class_names = df["class_id"].apply(
            lambda x: classes[int(x)] if pd.notna(x) else "background"
        )

        # Get counts for train and validation sets
        train_counts = class_names[df["dataset"] == "train"].value_counts()
        valid_counts = class_names[df["dataset"] == "valid"].value_counts()

        # Combine all class names to ensure we have all categories
        all_classes = pd.concat([train_counts, valid_counts]).index.unique()

        # Create a DataFrame with all classes and fill missing values with 0
        df_plot = pd.DataFrame(
            {
                "Train": [train_counts.get(cls, 0) for cls in all_classes],
                "Valid": [valid_counts.get(cls, 0) for cls in all_classes],
            },
            index=all_classes,
        )

        # Plot both distributions in one graph with different colors
        df_plot.plot(kind="bar", ax=ax, color=["blue", "orange"])

        plot_title = (
            title
            if title is not None
            else "Class Distribution in Training and Validation Sets"
        )
        ax.set_title(plot_title)
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")
        ax.legend(["Training Set", "Validation Set"])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_split_distribution(df: pd.DataFrame) -> None:
        """Plot the distribution of train/valid splits.

        Args:
            df: DataFrame containing the dataset
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        split_counts = df["dataset"].value_counts()
        split_counts.plot(kind="bar", ax=ax, color=["blue", "orange"])

        ax.set_title("Dataset Split Distribution")
        ax.set_ylabel("Number of Detections")
        ax.set_xlabel("Split")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_detections_per_image(df: pd.DataFrame) -> None:
        """Plot histogram of number of detections per image.

        Args:
            df: DataFrame containing the dataset
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Count detections per image
        detections_per_image = df.groupby("image_path").size()

        ax.hist(detections_per_image, bins=50, edgecolor="black")
        ax.set_title("Distribution of Detections per Image")
        ax.set_xlabel("Number of Detections")
        ax.set_ylabel("Number of Images")
        ax.axvline(
            detections_per_image.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {detections_per_image.mean():.2f}",
        )
        ax.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_bbox_size_distribution(df: pd.DataFrame) -> None:
        """Plot distribution of bounding box sizes.

        Args:
            df: DataFrame containing the dataset
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Extract widths and heights from xywh
        widths = []
        heights = []
        for xywh in df["xywh"].dropna():
            if len(xywh) >= 4:
                widths.append(xywh[2])
                heights.append(xywh[3])

        # Plot width distribution
        ax1.hist(widths, bins=50, edgecolor="black", alpha=0.7)
        ax1.set_title("Bounding Box Width Distribution")
        ax1.set_xlabel("Width (normalized)")
        ax1.set_ylabel("Count")
        ax1.axvline(
            sum(widths) / len(widths) if widths else 0,
            color="red",
            linestyle="--",
            label=f"Mean: {sum(widths) / len(widths):.3f}" if widths else "Mean: 0",
        )
        ax1.legend()

        # Plot height distribution
        ax2.hist(heights, bins=50, edgecolor="black", alpha=0.7, color="orange")
        ax2.set_title("Bounding Box Height Distribution")
        ax2.set_xlabel("Height (normalized)")
        ax2.set_ylabel("Count")
        ax2.axvline(
            sum(heights) / len(heights) if heights else 0,
            color="red",
            linestyle="--",
            label=f"Mean: {sum(heights) / len(heights):.3f}" if heights else "Mean: 0",
        )
        ax2.legend()

        plt.tight_layout()
        plt.show()
