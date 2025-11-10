"""Tools for comparing results from different experiments.

Enables A/B testing of prompts, models, and configurations.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from action_labeler.labeler.storage.label_store import LabelStore


@dataclass
class ComparisonResult:
    """Results from comparing two label stores.

    Attributes:
        total_detections: Total detections compared
        agreements: Number of detections with same label
        disagreements: Number of detections with different labels
        only_in_store1: Detections only labeled in store 1
        only_in_store2: Detections only labeled in store 2
        agreement_rate: Percentage agreement (0-1)
        disagreement_details: List of (image_path, xywh, label1, label2) tuples
    """

    total_detections: int
    agreements: int
    disagreements: int
    only_in_store1: int
    only_in_store2: int
    agreement_rate: float
    disagreement_details: list[tuple[str, list[float], str, str]]


class ExperimentComparator:
    """Compares results from different labeling experiments.

    Useful for:
    - A/B testing different prompts
    - Comparing different models
    - Identifying labeling inconsistencies
    - Finding low-confidence cases (disagreements between runs)
    """

    @staticmethod
    def compare_stores(
        store1: LabelStore,
        store2: LabelStore,
        include_details: bool = True,
    ) -> ComparisonResult:
        """Compare two label stores.

        Args:
            store1: First label store
            store2: Second label store
            include_details: Whether to include disagreement details

        Returns:
            ComparisonResult with comparison statistics
        """
        store1.flush()
        store2.flush()

        # Get DataFrames
        df1 = store1.to_simple_dataframe()
        df2 = store2.to_simple_dataframe()

        # Create detection keys for matching
        df1["det_key"] = df1.apply(
            lambda row: (row["image_path"], tuple(row["xywh"])), axis=1
        )
        df2["det_key"] = df2.apply(
            lambda row: (row["image_path"], tuple(row["xywh"])), axis=1
        )

        # Find common detections
        keys1 = set(df1["det_key"])
        keys2 = set(df2["det_key"])

        common_keys = keys1 & keys2
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1

        # Compare labels for common detections
        agreements = 0
        disagreements = 0
        disagreement_details = []

        for key in common_keys:
            label1 = df1[df1["det_key"] == key].iloc[0]["label"]
            label2 = df2[df2["det_key"] == key].iloc[0]["label"]

            if label1 == label2:
                agreements += 1
            else:
                disagreements += 1

                if include_details:
                    image_path, xywh = key
                    disagreement_details.append(
                        (image_path, list(xywh), label1, label2)
                    )

        total = len(common_keys)
        agreement_rate = agreements / total if total > 0 else 0.0

        return ComparisonResult(
            total_detections=total,
            agreements=agreements,
            disagreements=disagreements,
            only_in_store1=len(only_in_1),
            only_in_store2=len(only_in_2),
            agreement_rate=agreement_rate,
            disagreement_details=disagreement_details,
        )

    @staticmethod
    def find_disagreements(
        store1: LabelStore,
        store2: LabelStore,
        min_count: int = 1,
    ) -> pd.DataFrame:
        """Find detections where experiments disagree.

        Args:
            store1: First label store
            store2: Second label store
            min_count: Minimum number of disagreements to include

        Returns:
            DataFrame with disagreeing detections
        """
        result = ExperimentComparator.compare_stores(
            store1, store2, include_details=True
        )

        if not result.disagreement_details:
            return pd.DataFrame(
                columns=["image_path", "xywh", "label_store1", "label_store2"]
            )

        # Convert to DataFrame
        df = pd.DataFrame(
            result.disagreement_details,
            columns=["image_path", "xywh", "label_store1", "label_store2"],
        )

        # Filter by count if needed
        if min_count > 1:
            # Group by image and count disagreements
            image_counts = df["image_path"].value_counts()
            valid_images = image_counts[image_counts >= min_count].index
            df = df[df["image_path"].isin(valid_images)]

        return df

    @staticmethod
    def compute_confusion_matrix(
        store1: LabelStore,
        store2: LabelStore,
    ) -> pd.DataFrame:
        """Compute confusion matrix between two label stores.

        Args:
            store1: First label store (rows)
            store2: Second label store (columns)

        Returns:
            DataFrame confusion matrix
        """
        result = ExperimentComparator.compare_stores(
            store1, store2, include_details=True
        )

        if not result.disagreement_details:
            return pd.DataFrame()

        # Create confusion data
        confusion_data = []

        for image_path, xywh, label1, label2 in result.disagreement_details:
            confusion_data.append({"store1_label": label1, "store2_label": label2})

        # Also add agreements
        store1.flush()
        store2.flush()

        df1 = store1.to_simple_dataframe()
        df2 = store2.to_simple_dataframe()

        df1["det_key"] = df1.apply(
            lambda row: (row["image_path"], tuple(row["xywh"])), axis=1
        )
        df2["det_key"] = df2.apply(
            lambda row: (row["image_path"], tuple(row["xywh"])), axis=1
        )

        common_keys = set(df1["det_key"]) & set(df2["det_key"])

        for key in common_keys:
            label1 = df1[df1["det_key"] == key].iloc[0]["label"]
            label2 = df2[df2["det_key"] == key].iloc[0]["label"]

            if label1 == label2:
                confusion_data.append({"store1_label": label1, "store2_label": label2})

        # Create confusion matrix
        df = pd.DataFrame(confusion_data)
        confusion_matrix = pd.crosstab(
            df["store1_label"], df["store2_label"], margins=True
        )

        return confusion_matrix

    @staticmethod
    def get_label_agreement_by_class(
        store1: LabelStore,
        store2: LabelStore,
    ) -> dict[str, dict[str, Any]]:
        """Get agreement statistics per class.

        Args:
            store1: First label store
            store2: Second label store

        Returns:
            Dictionary mapping class names to agreement stats
        """
        result = ExperimentComparator.compare_stores(
            store1, store2, include_details=True
        )

        # Group by class
        class_stats: dict[str, dict[str, Any]] = {}

        # Initialize with all classes from both stores
        store1.flush()
        store2.flush()

        all_labels = set(store1.df["label"]) | set(store2.df["label"])

        for label in all_labels:
            class_stats[label] = {
                "agreements": 0,
                "disagreements": 0,
                "agreement_rate": 0.0,
            }

        # Count agreements/disagreements per class
        for image_path, xywh, label1, label2 in result.disagreement_details:
            # This is a disagreement
            if label1 in class_stats:
                class_stats[label1]["disagreements"] += 1
            if label2 in class_stats:
                class_stats[label2]["disagreements"] += 1

        # Count agreements
        df1 = store1.to_simple_dataframe()
        df2 = store2.to_simple_dataframe()

        df1["det_key"] = df1.apply(
            lambda row: (row["image_path"], tuple(row["xywh"])), axis=1
        )
        df2["det_key"] = df2.apply(
            lambda row: (row["image_path"], tuple(row["xywh"])), axis=1
        )

        common_keys = set(df1["det_key"]) & set(df2["det_key"])

        for key in common_keys:
            label1 = df1[df1["det_key"] == key].iloc[0]["label"]
            label2 = df2[df2["det_key"] == key].iloc[0]["label"]

            if label1 == label2:
                if label1 in class_stats:
                    class_stats[label1]["agreements"] += 1

        # Calculate rates
        for label, stats in class_stats.items():
            total = stats["agreements"] + stats["disagreements"]
            if total > 0:
                stats["agreement_rate"] = stats["agreements"] / total

        return class_stats

    @staticmethod
    def print_comparison(
        result: ComparisonResult,
        show_details: bool = False,
        max_details: int = 10,
    ) -> None:
        """Print comparison results to console.

        Args:
            result: Comparison result to print
            show_details: Whether to show disagreement details
            max_details: Maximum number of details to show
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPARISON")
        print("=" * 60)
        print(f"Total Detections Compared: {result.total_detections}")
        print(f"Agreements: {result.agreements} ({result.agreement_rate:.1%})")
        print(f"Disagreements: {result.disagreements} ({1-result.agreement_rate:.1%})")
        print(f"Only in Store 1: {result.only_in_store1}")
        print(f"Only in Store 2: {result.only_in_store2}")
        print("=" * 60)

        if show_details and result.disagreement_details:
            print(f"\nDisagreement Details (showing first {max_details}):")
            print("-" * 60)

            for i, (img_path, xywh, label1, label2) in enumerate(
                result.disagreement_details[:max_details]
            ):
                from pathlib import Path

                img_name = Path(img_path).name
                print(f"{i+1}. {img_name}")
                print(f"   Store 1: {label1}")
                print(f"   Store 2: {label2}")
                print(f"   bbox: {xywh}")

            if len(result.disagreement_details) > max_details:
                remaining = len(result.disagreement_details) - max_details
                print(f"\n... and {remaining} more disagreements")

        print()
