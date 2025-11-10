"""Statistical analysis and visualization tools for labeled datasets."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.helpers.image_helpers import add_bounding_boxes, add_text
from action_labeler.labeler.storage.label_store import LabelStore


class DatasetAnalyzer:
    """Analyzes labeled datasets for quality, distribution, and patterns.

    Provides:
    - Label distribution statistics
    - Quality metrics (invalid labels, confidence scores)
    - Visualization tools
    - Per-experiment analysis
    """

    def __init__(self, label_store: LabelStore):
        """Initialize analyzer with a label store.

        Args:
            label_store: Store to analyze
        """
        self.label_store = label_store
        self.label_store.flush()

    def get_label_distribution(self) -> dict[str, int]:
        """Get distribution of labels.

        Returns:
            Dictionary mapping labels to counts
        """
        return self.label_store.df["label"].value_counts().to_dict()

    def get_experiment_distribution(self) -> dict[str, int]:
        """Get distribution of labels by experiment.

        Returns:
            Dictionary mapping experiment IDs to counts
        """
        from action_labeler.labeler.storage.metadata import LabelMetadata

        experiment_counts: dict[str, int] = {}

        for _, row in self.label_store.df.iterrows():
            metadata = LabelMetadata.from_dict(row["metadata"])
            exp_id = metadata.experiment_id
            experiment_counts[exp_id] = experiment_counts.get(exp_id, 0) + 1

        return experiment_counts

    def get_confidence_statistics(self) -> dict[str, Any]:
        """Get statistics about confidence scores.

        Returns:
            Dictionary with confidence statistics
        """
        from action_labeler.labeler.storage.metadata import LabelMetadata

        confidences = []

        for _, row in self.label_store.df.iterrows():
            metadata = LabelMetadata.from_dict(row["metadata"])
            if metadata.confidence is not None:
                confidences.append(metadata.confidence)

        if not confidences:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
            }

        return {
            "count": len(confidences),
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "median": np.median(confidences),
        }

    def get_invalid_labels(self) -> list[dict[str, Any]]:
        """Get all invalid labels (failed validation).

        Returns:
            List of invalid label records
        """
        from action_labeler.labeler.storage.metadata import LabelMetadata

        invalid_labels = []

        for _, row in self.label_store.df.iterrows():
            metadata = LabelMetadata.from_dict(row["metadata"])
            if not metadata.is_valid:
                invalid_labels.append(
                    {
                        "image_path": row["image_path"],
                        "xywh": row["xywh"],
                        "label": row["label"],
                        "error": metadata.validation_error,
                        "experiment_id": metadata.experiment_id,
                    }
                )

        return invalid_labels

    def get_labels_by_confidence_range(
        self, min_conf: float, max_conf: float
    ) -> pd.DataFrame:
        """Get labels within a confidence range.

        Useful for finding low-confidence predictions to review.

        Args:
            min_conf: Minimum confidence (inclusive)
            max_conf: Maximum confidence (inclusive)

        Returns:
            DataFrame with labels in range
        """
        from action_labeler.labeler.storage.metadata import LabelMetadata

        filtered_rows = []

        for _, row in self.label_store.df.iterrows():
            metadata = LabelMetadata.from_dict(row["metadata"])

            if metadata.confidence is not None:
                if min_conf <= metadata.confidence <= max_conf:
                    filtered_rows.append(row.to_dict())

        return pd.DataFrame(filtered_rows)

    def plot_label_distribution(
        self, title: str | None = None, figsize: tuple = (12, 6)
    ) -> None:
        """Plot distribution of labels.

        Args:
            title: Custom title for plot
            figsize: Figure size (width, height)
        """
        distribution = self.get_label_distribution()

        if not distribution:
            print("No labels to plot")
            return

        fig, ax = plt.subplots(figsize=figsize)

        labels = list(distribution.keys())
        counts = list(distribution.values())

        ax.bar(labels, counts, color="skyblue", edgecolor="black")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_title(title or "Label Distribution")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_confidence_distribution(
        self, bins: int = 50, figsize: tuple = (10, 6)
    ) -> None:
        """Plot distribution of confidence scores.

        Args:
            bins: Number of bins for histogram
            figsize: Figure size
        """
        from action_labeler.labeler.storage.metadata import LabelMetadata

        confidences = []

        for _, row in self.label_store.df.iterrows():
            metadata = LabelMetadata.from_dict(row["metadata"])
            if metadata.confidence is not None:
                confidences.append(metadata.confidence)

        if not confidences:
            print("No confidence scores available")
            return

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(confidences, bins=bins, color="green", alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(confidences),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(confidences):.3f}",
        )
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Score Distribution")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_images_for_label(
        self,
        label: str,
        num_samples: int = 5,
        figsize: tuple = (15, 10),
    ) -> None:
        """Plot sample images for a specific label.

        Args:
            label: Label to visualize
            num_samples: Number of sample images to show
            figsize: Figure size
        """
        # Get detections with this label
        label_df = self.label_store.df[self.label_store.df["label"] == label]

        if len(label_df) == 0:
            print(f"No detections found with label '{label}'")
            return

        # Sample images
        unique_images = label_df["image_path"].unique()
        num_to_show = min(num_samples, len(unique_images))
        sampled_images = np.random.choice(unique_images, num_to_show, replace=False)

        # Create grid
        cols = min(3, num_to_show)
        rows = (num_to_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_to_show == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_to_show > 1 else axes

        for idx, image_path in enumerate(sampled_images):
            # Get all detections for this image with this label
            image_detections = label_df[label_df["image_path"] == image_path]

            # Load image
            try:
                img = Image.open(image_path)
                img_width, img_height = img.size

                # Draw bounding boxes
                for _, row in image_detections.iterrows():
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

                    # Create Detection for drawing
                    detection = Detection(
                        xyxy=np.array([[x1, y1, x2, y2]]),
                        segmentation_points=[[]],
                        keypoints=np.array([]),
                        class_id=np.array([0]),
                        image=img,
                    )

                    img = add_bounding_boxes(img, detection, color="red", width=3)
                    img = add_text(
                        img,
                        0,
                        detection,
                        label,
                        text_color="red",
                        font_size=20,
                    )

                axes[idx].imshow(img)
                axes[idx].axis("off")
                axes[idx].set_title(Path(image_path).name, fontsize=10)

            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                axes[idx].axis("off")

        # Hide unused subplots
        for idx in range(num_to_show, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"Sample Images for Label '{label}' ({num_to_show} of {len(unique_images)})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive analysis report.

        Returns:
            Dictionary with analysis results
        """
        stats = self.label_store.get_statistics()
        confidence_stats = self.get_confidence_statistics()
        invalid_labels = self.get_invalid_labels()
        experiment_dist = self.get_experiment_distribution()

        return {
            "total_labels": stats["total_labels"],
            "unique_images": stats["unique_images"],
            "unique_labels": stats["unique_labels"],
            "label_distribution": stats["label_distribution"],
            "experiment_distribution": experiment_dist,
            "invalid_count": stats["invalid_count"],
            "invalid_labels": invalid_labels,
            "confidence_statistics": confidence_stats,
        }

    def print_report(self) -> None:
        """Print comprehensive analysis report to console."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("DATASET ANALYSIS REPORT")
        print("=" * 60)
        print(f"Total Labels: {report['total_labels']}")
        print(f"Unique Images: {report['unique_images']}")
        print(f"Unique Label Classes: {report['unique_labels']}")
        print(f"Invalid Labels: {report['invalid_count']}")
        print("=" * 60)

        print("\nLabel Distribution:")
        for label, count in sorted(
            report["label_distribution"].items(), key=lambda x: -x[1]
        ):
            percentage = (count / report["total_labels"]) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")

        print("\nExperiment Distribution:")
        for exp_id, count in sorted(
            report["experiment_distribution"].items(), key=lambda x: -x[1]
        ):
            percentage = (count / report["total_labels"]) * 100
            # Shorten exp_id for display
            short_id = exp_id[:16] + "..." if len(exp_id) > 16 else exp_id
            print(f"  {short_id}: {count} ({percentage:.1f}%)")

        if report["confidence_statistics"]["count"] > 0:
            conf = report["confidence_statistics"]
            print("\nConfidence Statistics:")
            print(f"  Count: {conf['count']}")
            print(f"  Mean: {conf['mean']:.3f}")
            print(f"  Std Dev: {conf['std']:.3f}")
            print(f"  Min: {conf['min']:.3f}")
            print(f"  Max: {conf['max']:.3f}")
            print(f"  Median: {conf['median']:.3f}")

        if report["invalid_labels"]:
            print(f"\nInvalid Labels ({len(report['invalid_labels'])}):")
            for i, invalid in enumerate(report["invalid_labels"][:5]):
                print(f"  {i+1}. {Path(invalid['image_path']).name}")
                print(f"     Label: {invalid['label']}")
                print(f"     Error: {invalid['error']}")

            if len(report["invalid_labels"]) > 5:
                print(f"  ... and {len(report['invalid_labels']) - 5} more")

        print("=" * 60)
        print()
