"""Visualization operations for YoloV8Dataset.

This module handles plotting and visualization of dataset information.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import add_bounding_boxes, add_text


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

    @staticmethod
    def plot_class_samples(
        df: pd.DataFrame,
        classes: list[str],
        class_name: str,
        class_id: int,
        num_samples: int = 5,
    ) -> None:
        """Plot sample images for a specific class with bounding boxes.

        Args:
            df: DataFrame containing the dataset
            classes: List of class names
            class_name: Name of the class to visualize
            class_id: ID of the class to visualize
            num_samples: Number of sample images to show (default: 5)
        """
        # Filter dataframe to only include rows with the target class
        class_df = df[df["class_id"] == class_id].copy()

        if len(class_df) == 0:
            print(f"No samples found for class '{class_name}'")
            return

        # Get unique images that contain this class
        unique_images = class_df["image_path"].unique()

        # Sample up to num_samples images randomly
        num_to_show = min(num_samples, len(unique_images))
        sampled_images = np.random.choice(unique_images, num_to_show, replace=False)

        # Calculate grid layout
        cols = min(3, num_to_show)  # Max 3 columns
        rows = (num_to_show + cols - 1) // cols  # Ceiling division

        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

        # Handle case where there's only one subplot
        if num_to_show == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_to_show > 1 else axes

        # Plot each sampled image
        for idx, image_path in enumerate(sampled_images):
            # Get all detections for this image
            image_detections = df[df["image_path"] == image_path]

            # Load the image
            img = Image.open(image_path)
            img_width, img_height = img.size

            # Convert normalized xywh to pixel xyxy for all detections
            xyxy_list = []
            class_ids = []
            for _, row in image_detections.iterrows():
                if pd.notna(row["xywh"]):
                    xywh = row["xywh"]
                    # Convert normalized xywh to pixel xyxy
                    x_center = xywh[0] * img_width
                    y_center = xywh[1] * img_height
                    width = xywh[2] * img_width
                    height = xywh[3] * img_height

                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    xyxy_list.append([x1, y1, x2, y2])
                    class_ids.append(int(row["class_id"]))

            # Create Detection object
            if xyxy_list:
                detection = Detection(
                    xyxy=np.array(xyxy_list),
                    segmentation_points=[[] for _ in xyxy_list],  # Empty for bbox
                    keypoints=np.array([]),  # Empty keypoints
                    class_id=np.array(class_ids),
                    image=img,
                )

                # Draw bounding boxes - highlight target class in red, others in blue
                for det_idx in range(len(detection.xyxy)):
                    if detection.class_id[det_idx] == class_id:
                        # Target class in red with thicker line
                        img = add_bounding_boxes(
                            img,
                            Detection(
                                xyxy=np.array([detection.xyxy[det_idx]]),
                                segmentation_points=[[]],
                                keypoints=np.array([]),
                                class_id=np.array([detection.class_id[det_idx]]),
                                image=img,
                            ),
                            color="red",
                            width=4,
                        )
                        # Add class label
                        img = add_text(
                            img,
                            0,
                            Detection(
                                xyxy=np.array([detection.xyxy[det_idx]]),
                                segmentation_points=[[]],
                                keypoints=np.array([]),
                                class_id=np.array([detection.class_id[det_idx]]),
                                image=img,
                            ),
                            classes[int(detection.class_id[det_idx])],
                            text_color="red",
                            font_size=20,
                        )
                    else:
                        # Other classes in blue with normal line
                        img = add_bounding_boxes(
                            img,
                            Detection(
                                xyxy=np.array([detection.xyxy[det_idx]]),
                                segmentation_points=[[]],
                                keypoints=np.array([]),
                                class_id=np.array([detection.class_id[det_idx]]),
                                image=img,
                            ),
                            color="blue",
                            width=2,
                        )

            # Display the image
            axes[idx].imshow(img)
            axes[idx].axis("off")
            image_name = Path(image_path).name
            axes[idx].set_title(f"{image_name}", fontsize=10)

        # Hide any unused subplots
        for idx in range(num_to_show, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"Sample Images for Class '{class_name}' ({num_to_show} of {len(unique_images)} images)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
