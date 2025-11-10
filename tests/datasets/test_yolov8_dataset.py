"""Comprehensive tests for YoloV8Dataset class.

This module contains extensive pytest tests for the YoloV8Dataset class,
covering all major functionality including loading, saving, transformations,
validation, and visualization.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.datasets.helpers import (
    assert_class_distribution,
    assert_dataset_equal,
    assert_split_ratio,
    cleanup_temp_dataset,
    create_empty_dataset_folder,
    create_temp_yolo_dataset,
)

from action_labeler.dataclasses import DetectionType
from action_labeler.datasets import YoloV8Dataset
from action_labeler.datasets.dataset_config import DatasetConfig
from action_labeler.datasets.exceptions import (
    ClassMappingError,
    ClassNotFoundError,
    DatasetIOError,
    DatasetValidationError,
    EmptyDatasetError,
    InvalidSplitError,
)


class TestYoloV8DatasetConstructor:
    """Test cases for YoloV8Dataset constructor validation."""

    def test_constructor_with_valid_inputs(self, temp_dir, sample_classes):
        """Test that constructor accepts valid inputs."""
        df = pd.DataFrame(columns=["dataset", "image_path", "xywh", "class_id"])
        dataset = YoloV8Dataset(temp_dir, sample_classes, df)

        assert dataset.folder == temp_dir
        assert dataset.classes == sample_classes
        assert len(dataset.df) == 0
        assert dataset.detection_type == DetectionType.DETECT

    def test_constructor_with_custom_config(self, temp_dir, sample_classes):
        """Test constructor with custom configuration."""
        config = DatasetConfig(train_split=0.7, valid_split=0.3, random_seed=123)
        df = pd.DataFrame(columns=["dataset", "image_path", "xywh", "class_id"])
        dataset = YoloV8Dataset(temp_dir, sample_classes, df, config=config)

        assert dataset.config.train_split == 0.7
        assert dataset.config.valid_split == 0.3
        assert dataset.config.random_seed == 123

    def test_constructor_with_empty_classes_raises_error(self, temp_dir):
        """Test that constructor raises error with empty classes list."""
        df = pd.DataFrame(columns=["dataset", "image_path", "xywh", "class_id"])
        with pytest.raises(DatasetValidationError):
            YoloV8Dataset(temp_dir, [], df)

    def test_constructor_with_duplicate_classes_raises_error(self, temp_dir):
        """Test that constructor raises error with duplicate class names."""
        df = pd.DataFrame(columns=["dataset", "image_path", "xywh", "class_id"])
        with pytest.raises(DatasetValidationError):
            YoloV8Dataset(temp_dir, ["dog", "cat", "dog"], df)

    def test_constructor_creates_class_name_to_id_mapping(
        self, temp_dir, sample_classes
    ):
        """Test that constructor creates correct class name to ID mapping."""
        df = pd.DataFrame(columns=["dataset", "image_path", "xywh", "class_id"])
        dataset = YoloV8Dataset(temp_dir, sample_classes, df)

        assert dataset.class_name_to_id == {
            "dog": 0,
            "cat": 1,
            "bird": 2,
            "fish": 3,
        }

    def test_constructor_with_segment_detection_type(self, temp_dir, sample_classes):
        """Test constructor with segment detection type."""
        df = pd.DataFrame(columns=["dataset", "image_path", "xywh", "class_id"])
        dataset = YoloV8Dataset(
            temp_dir, sample_classes, df, detection_type=DetectionType.SEGMENT
        )

        assert dataset.detection_type == DetectionType.SEGMENT


class TestYoloV8DatasetLoading:
    """Test cases for loading datasets from disk."""

    def test_from_folder_loads_valid_dataset(self, sample_dataset):
        """Test that from_folder loads a valid dataset."""
        assert len(sample_dataset) > 0
        assert len(sample_dataset.classes) == 4
        assert "train" in sample_dataset.df["dataset"].values
        assert "valid" in sample_dataset.df["dataset"].values

    def test_from_folder_with_nonexistent_folder_raises_error(self):
        """Test that from_folder raises error for nonexistent folder."""
        with pytest.raises(DatasetIOError):
            YoloV8Dataset.from_folder("/nonexistent/path/to/dataset")

    def test_from_folder_loads_correct_class_names(self, sample_dataset):
        """Test that from_folder loads correct class names."""
        assert sample_dataset.classes == ["dog", "cat", "bird", "fish"]

    def test_from_folder_creates_correct_dataframe_structure(self, sample_dataset):
        """Test that from_folder creates correct DataFrame structure."""
        required_columns = {"dataset", "image_path", "xywh", "class_id"}
        assert required_columns.issubset(set(sample_dataset.df.columns))

    def test_from_folder_with_custom_config(self, temp_dir, sample_classes):
        """Test from_folder with custom configuration."""
        temp_folder, _ = create_temp_yolo_dataset(sample_classes)
        config = DatasetConfig(random_seed=999)

        try:
            dataset = YoloV8Dataset.from_folder(temp_folder, config=config)
            assert dataset.config.random_seed == 999
        finally:
            cleanup_temp_dataset(temp_folder)

    def test_empty_creates_empty_dataset(self, temp_dir, sample_classes):
        """Test that empty() creates a valid empty dataset."""
        dataset = YoloV8Dataset.empty(temp_dir, sample_classes)

        assert len(dataset) == 0
        assert dataset.classes == sample_classes
        assert dataset.folder == temp_dir


class TestYoloV8DatasetSaving:
    """Test cases for saving datasets to disk."""

    def test_save_creates_directory_structure(self, sample_dataset, temp_dir):
        """Test that save() creates correct directory structure."""
        output_folder = temp_dir / "output"
        sample_dataset.save(output_folder, delete_existing=True)

        assert (output_folder / "train" / "images").exists()
        assert (output_folder / "train" / "labels").exists()
        assert (output_folder / "valid" / "images").exists()
        assert (output_folder / "valid" / "labels").exists()
        assert (output_folder / "data.yaml").exists()

    def test_save_and_load_roundtrip(self, sample_dataset, temp_dir):
        """Test that save and load roundtrip preserves data."""
        output_folder = temp_dir / "output"
        sample_dataset.save(output_folder, delete_existing=True)

        # Load the saved dataset
        loaded_dataset = YoloV8Dataset.from_folder(output_folder)

        # Check that key properties match
        assert loaded_dataset.classes == sample_dataset.classes
        assert len(loaded_dataset) == len(sample_dataset)

    def test_save_with_delete_existing(self, sample_dataset, temp_dir):
        """Test that save with delete_existing removes old files."""
        output_folder = temp_dir / "output"

        # Create a marker file
        output_folder.mkdir(parents=True, exist_ok=True)
        marker_file = output_folder / "marker.txt"
        marker_file.write_text("test")

        # Save with delete_existing=True
        sample_dataset.save(output_folder, delete_existing=True)

        # Marker file should be gone
        assert not marker_file.exists()

    def test_save_returns_self_for_chaining(self, sample_dataset, temp_dir):
        """Test that save() returns self for method chaining."""
        output_folder = temp_dir / "output"
        result = sample_dataset.save(output_folder, delete_existing=True)

        assert result is sample_dataset


class TestYoloV8DatasetClassOperations:
    """Test cases for class manipulation operations."""

    def test_remap_classes_changes_class_names(self, sample_dataset):
        """Test that remap_classes changes class names correctly."""
        original_len = len(sample_dataset)

        sample_dataset.remap_classes({"dog": "canine", "cat": "feline"})

        assert "canine" in sample_dataset.classes
        assert "feline" in sample_dataset.classes
        assert "dog" not in sample_dataset.classes
        assert "cat" not in sample_dataset.classes
        assert len(sample_dataset) == original_len  # Data preserved

    def test_remap_classes_with_nonexistent_class_raises_error(self, sample_dataset):
        """Test that remapping nonexistent class raises error."""
        with pytest.raises(ClassNotFoundError):
            sample_dataset.remap_classes({"unicorn": "mythical"})

    def test_remap_classes_returns_self(self, sample_dataset):
        """Test that remap_classes returns self for chaining."""
        result = sample_dataset.remap_classes({"dog": "canine"})
        assert result is sample_dataset

    def test_delete_classes_removes_class_data(self, sample_dataset):
        """Test that delete_classes removes class and its data."""
        original_classes = len(sample_dataset.classes)
        original_len = len(sample_dataset)

        sample_dataset.delete_classes(["fish"])

        assert "fish" not in sample_dataset.classes
        assert len(sample_dataset.classes) == original_classes - 1
        assert len(sample_dataset) < original_len  # Some data removed

    def test_delete_classes_updates_class_ids_sequentially(self, sample_dataset):
        """Test that delete_classes maintains sequential class IDs."""
        sample_dataset.delete_classes(["cat"])

        # Class IDs should be 0, 1, 2 (no gaps)
        unique_class_ids = sorted(sample_dataset.df["class_id"].unique())
        expected_ids = list(range(len(sample_dataset.classes)))

        assert unique_class_ids == expected_ids

    def test_delete_multiple_classes(self, sample_dataset):
        """Test deleting multiple classes at once."""
        sample_dataset.delete_classes(["dog", "cat"])

        assert "dog" not in sample_dataset.classes
        assert "cat" not in sample_dataset.classes
        assert len(sample_dataset.classes) == 2

    def test_delete_nonexistent_class_raises_error(self, sample_dataset):
        """Test that deleting nonexistent class raises error."""
        with pytest.raises(ClassNotFoundError):
            sample_dataset.delete_classes(["unicorn"])

    def test_delete_classes_returns_self(self, sample_dataset):
        """Test that delete_classes returns self for chaining."""
        result = sample_dataset.delete_classes(["fish"])
        assert result is sample_dataset


class TestYoloV8DatasetBalancing:
    """Test cases for dataset balancing operations."""

    def test_create_balanced_dataset_returns_new_instance(self, unbalanced_dataset):
        """Test that create_balanced_dataset returns a new instance."""
        balanced = unbalanced_dataset.create_balanced_dataset()

        assert balanced is not unbalanced_dataset
        assert isinstance(balanced, YoloV8Dataset)

    def test_create_balanced_dataset_balances_classes(self, unbalanced_dataset):
        """Test that balancing creates equal class distribution."""
        balanced = unbalanced_dataset.create_balanced_dataset()

        # All classes should have the same count (minimum count from original)
        class_counts = balanced.df["class_id"].value_counts()
        assert len(class_counts.unique()) == 1  # All counts are equal

    def test_create_balanced_dataset_with_min_samples(self, unbalanced_dataset):
        """Test balancing with explicit min_samples."""
        balanced = unbalanced_dataset.create_balanced_dataset(min_samples=5)

        # All classes should have exactly 5 samples
        class_counts = balanced.df["class_id"].value_counts()
        assert all(count == 5 for count in class_counts)

    def test_create_balanced_dataset_with_random_state(self, unbalanced_dataset):
        """Test that random_state makes balancing reproducible."""
        balanced1 = unbalanced_dataset.create_balanced_dataset(random_state=42)
        balanced2 = unbalanced_dataset.create_balanced_dataset(random_state=42)

        # Should get the same samples
        assert len(balanced1) == len(balanced2)

    def test_create_balanced_dataset_assigns_train_valid_split(
        self, unbalanced_dataset
    ):
        """Test that balanced dataset has train/valid splits."""
        balanced = unbalanced_dataset.create_balanced_dataset()

        assert "train" in balanced.df["dataset"].values
        assert "valid" in balanced.df["dataset"].values

    def test_create_balanced_dataset_on_empty_raises_error(self, empty_dataset):
        """Test that balancing empty dataset raises error."""
        with pytest.raises(EmptyDatasetError):
            empty_dataset.create_balanced_dataset()

    def test_add_background_images_adds_images(self, sample_dataset, temp_dir):
        """Test that add_background_images adds background images."""
        # Create background images
        bg_folder = temp_dir / "backgrounds"
        bg_folder.mkdir(exist_ok=True)
        for i in range(10):
            (bg_folder / f"bg_{i}.jpg").write_text("background")

        original_len = len(sample_dataset)
        sample_dataset.add_background_images(bg_folder, pct_background=0.2)

        # Should have added some background images
        assert len(sample_dataset) > original_len

    def test_add_background_images_returns_self(self, sample_dataset, temp_dir):
        """Test that add_background_images returns self."""
        bg_folder = temp_dir / "backgrounds"
        bg_folder.mkdir(exist_ok=True)
        (bg_folder / "bg_1.jpg").write_text("background")

        result = sample_dataset.add_background_images(bg_folder)
        assert result is sample_dataset

    def test_add_background_images_on_empty_raises_error(self, empty_dataset, temp_dir):
        """Test that adding background to empty dataset raises error."""
        bg_folder = temp_dir / "backgrounds"
        bg_folder.mkdir(exist_ok=True)

        with pytest.raises(EmptyDatasetError):
            empty_dataset.add_background_images(bg_folder)


class TestYoloV8DatasetMerging:
    """Test cases for merging datasets."""

    def test_merge_with_union_combines_all_classes(
        self, sample_dataset, two_class_dataset
    ):
        """Test that merge with union strategy combines all classes."""
        merged = sample_dataset.merge(two_class_dataset, strategy="union")

        # Should have all unique classes from both datasets
        all_classes = set(sample_dataset.classes + two_class_dataset.classes)
        assert set(merged.classes) == all_classes

    def test_merge_with_union_combines_all_data(
        self, sample_dataset, two_class_dataset
    ):
        """Test that merge combines all detection data."""
        original_len1 = len(sample_dataset)
        original_len2 = len(two_class_dataset)

        merged = sample_dataset.merge(two_class_dataset, strategy="union")

        assert len(merged) == original_len1 + original_len2

    def test_merge_with_intersection_keeps_only_common_classes(self, sample_dataset):
        """Test that merge with intersection keeps only common classes."""
        # Create another dataset with overlapping classes
        temp_folder, other_dataset = create_temp_yolo_dataset(
            classes=["dog", "cat", "horse"],  # dog and cat overlap
            num_images_per_split={"train": 5, "valid": 2},
        )

        try:
            merged = sample_dataset.merge(other_dataset, strategy="intersection")

            # Should only have common classes
            assert set(merged.classes) == {"dog", "cat"}
        finally:
            cleanup_temp_dataset(temp_folder)

    def test_merge_with_no_common_classes_raises_error(self, sample_dataset):
        """Test that merging with no common classes raises error."""
        temp_folder, other_dataset = create_temp_yolo_dataset(
            classes=["horse", "cow"],  # No overlap
            num_images_per_split={"train": 3, "valid": 1},
        )

        try:
            with pytest.raises(DatasetValidationError):
                sample_dataset.merge(other_dataset, strategy="intersection")
        finally:
            cleanup_temp_dataset(temp_folder)

    def test_merge_returns_new_instance(self, sample_dataset, two_class_dataset):
        """Test that merge returns a new dataset instance."""
        merged = sample_dataset.merge(two_class_dataset)

        assert merged is not sample_dataset
        assert merged is not two_class_dataset


class TestYoloV8DatasetFiltering:
    """Test cases for filtering datasets."""

    def test_filter_by_split_train(self, sample_dataset):
        """Test filtering to training split only."""
        train_only = sample_dataset.filter_by_split("train")

        assert all(train_only.df["dataset"] == "train")
        assert "valid" not in train_only.df["dataset"].values

    def test_filter_by_split_valid(self, sample_dataset):
        """Test filtering to validation split only."""
        valid_only = sample_dataset.filter_by_split("valid")

        assert all(valid_only.df["dataset"] == "valid")
        assert "train" not in valid_only.df["dataset"].values

    def test_filter_by_split_invalid_raises_error(self, sample_dataset):
        """Test that invalid split name raises error."""
        with pytest.raises(InvalidSplitError):
            sample_dataset.filter_by_split("test")

    def test_filter_by_split_returns_new_instance(self, sample_dataset):
        """Test that filter_by_split returns new instance."""
        filtered = sample_dataset.filter_by_split("train")

        assert filtered is not sample_dataset

    def test_filter_by_classes_keeps_only_specified_classes(self, sample_dataset):
        """Test that filter_by_classes keeps only specified classes."""
        filtered = sample_dataset.filter_by_classes(["dog", "cat"])

        assert filtered.classes == ["dog", "cat"]
        # All class IDs should be 0 or 1
        assert all(filtered.df["class_id"].dropna().isin([0, 1]))

    def test_filter_by_classes_with_nonexistent_class_raises_error(
        self, sample_dataset
    ):
        """Test that filtering by nonexistent class raises error."""
        with pytest.raises(ClassNotFoundError):
            sample_dataset.filter_by_classes(["dog", "unicorn"])

    def test_filter_by_classes_returns_new_instance(self, sample_dataset):
        """Test that filter_by_classes returns new instance."""
        filtered = sample_dataset.filter_by_classes(["dog"])

        assert filtered is not sample_dataset


class TestYoloV8DatasetValidation:
    """Test cases for dataset validation."""

    def test_validate_on_valid_dataset_succeeds(self, sample_dataset):
        """Test that validation succeeds on valid dataset."""
        # Don't check files exist since we have dummy files
        result = sample_dataset.validate(check_files_exist=False)

        assert result.is_valid

    def test_validate_detects_empty_dataset(self, empty_dataset):
        """Test that validation handles empty dataset."""
        result = empty_dataset.validate()

        # Empty dataset should be valid but with a warning
        assert result.is_valid
        assert any("empty" in w.lower() for w in result.warnings)

    def test_validate_returns_validation_result(self, sample_dataset):
        """Test that validate returns ValidationResult object."""
        from action_labeler.datasets.dataset_config import ValidationResult

        result = sample_dataset.validate(check_files_exist=False)

        assert isinstance(result, ValidationResult)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")


class TestYoloV8DatasetStatistics:
    """Test cases for dataset statistics."""

    def test_stats_property_returns_statistics(self, sample_dataset):
        """Test that stats property returns DatasetStats object."""
        from action_labeler.datasets.dataset_config import DatasetStats

        stats = sample_dataset.stats

        assert isinstance(stats, DatasetStats)

    def test_stats_num_images_correct(self, sample_dataset):
        """Test that stats reports correct number of images."""
        stats = sample_dataset.stats
        unique_images = sample_dataset.df["image_path"].nunique()

        assert stats.num_images == unique_images

    def test_stats_num_detections_correct(self, sample_dataset):
        """Test that stats reports correct number of detections."""
        stats = sample_dataset.stats

        assert stats.num_detections == len(sample_dataset)

    def test_stats_num_classes_correct(self, sample_dataset):
        """Test that stats reports correct number of classes."""
        stats = sample_dataset.stats

        assert stats.num_classes == len(sample_dataset.classes)

    def test_stats_class_distribution_correct(self, sample_dataset):
        """Test that stats reports correct class distribution."""
        stats = sample_dataset.stats

        # Check that all classes are present
        for class_name in sample_dataset.classes:
            assert class_name in stats.class_distribution

    def test_stats_split_counts_correct(self, sample_dataset):
        """Test that stats reports correct split counts."""
        stats = sample_dataset.stats

        train_count = len(sample_dataset.df[sample_dataset.df["dataset"] == "train"])
        valid_count = len(sample_dataset.df[sample_dataset.df["dataset"] == "valid"])

        assert stats.num_train_detections == train_count
        assert stats.num_valid_detections == valid_count

    def test_stats_caching(self, sample_dataset):
        """Test that stats are cached for performance."""
        stats1 = sample_dataset.stats
        stats2 = sample_dataset.stats

        # Should return the same object (cached)
        assert stats1 is stats2

    def test_stats_cache_invalidated_after_mutation(self, sample_dataset):
        """Test that stats cache is invalidated after dataset mutation."""
        stats1 = sample_dataset.stats

        # Mutate the dataset
        sample_dataset.delete_classes(["fish"])

        stats2 = sample_dataset.stats

        # Should be a different object (cache invalidated)
        assert stats1 is not stats2


class TestYoloV8DatasetVisualization:
    """Test cases for visualization methods."""

    def test_plot_class_distribution_returns_self(self, sample_dataset):
        """Test that plot_class_distribution returns self for chaining."""
        # Mock plt.show() to avoid display
        import matplotlib.pyplot as plt

        plt.ioff()  # Turn off interactive mode

        result = sample_dataset.plot_class_distribution()

        assert result is sample_dataset
        plt.close("all")

    def test_plot_split_distribution_returns_self(self, sample_dataset):
        """Test that plot_split_distribution returns self."""
        import matplotlib.pyplot as plt

        plt.ioff()

        result = sample_dataset.plot_split_distribution()

        assert result is sample_dataset
        plt.close("all")

    def test_plot_detections_per_image_returns_self(self, sample_dataset):
        """Test that plot_detections_per_image returns self."""
        import matplotlib.pyplot as plt

        plt.ioff()

        result = sample_dataset.plot_detections_per_image()

        assert result is sample_dataset
        plt.close("all")

    def test_plot_bbox_size_distribution_returns_self(self, sample_dataset):
        """Test that plot_bbox_size_distribution returns self."""
        import matplotlib.pyplot as plt

        plt.ioff()

        result = sample_dataset.plot_bbox_size_distribution()

        assert result is sample_dataset
        plt.close("all")

    def test_plot_dataset_returns_self(self, sample_dataset):
        """Test that plot_dataset returns self."""
        import matplotlib.pyplot as plt

        plt.ioff()

        result = sample_dataset.plot_dataset()

        assert result is sample_dataset
        plt.close("all")

    def test_plot_class_returns_self(self, sample_dataset, mocker):
        """Test that plot_class returns self for method chaining."""
        import matplotlib.pyplot as plt

        # Mock the visualizer method to avoid needing real images
        mock_plot = mocker.patch(
            "action_labeler.datasets.yolov8_dataset.YoloV8DatasetVisualizer.plot_class_samples"
        )

        result = sample_dataset.plot_class("dog", num_samples=2)

        # Verify the visualizer was called with correct arguments
        mock_plot.assert_called_once_with(
            sample_dataset.df,
            sample_dataset.classes,
            "dog",
            0,  # dog class_id
            2,
        )
        assert result is sample_dataset
        plt.close("all")

    def test_plot_class_raises_error_for_invalid_class(self, sample_dataset):
        """Test that plot_class raises ClassNotFoundError for invalid class."""
        with pytest.raises(ClassNotFoundError):
            sample_dataset.plot_class("invalid_class")


class TestYoloV8DatasetUtility:
    """Test cases for utility methods."""

    def test_len_returns_correct_count(self, sample_dataset):
        """Test that len() returns correct detection count."""
        expected_len = len(sample_dataset.df)

        assert len(sample_dataset) == expected_len

    def test_len_on_empty_dataset(self, empty_dataset):
        """Test that len() works on empty dataset."""
        assert len(empty_dataset) == 0

    def test_repr_contains_key_information(self, sample_dataset):
        """Test that __repr__ contains key dataset information."""
        repr_str = repr(sample_dataset)

        assert "YoloV8Dataset" in repr_str
        assert "classes" in repr_str
        assert "images" in repr_str
        assert "detections" in repr_str

    def test_copy_creates_independent_instance(self, sample_dataset):
        """Test that copy() creates an independent copy."""
        copy = sample_dataset.copy()

        # Should be different objects
        assert copy is not sample_dataset

        # But with same data
        assert copy.classes == sample_dataset.classes
        assert len(copy) == len(sample_dataset)

        # Modifying copy shouldn't affect original
        copy.delete_classes(["fish"])
        assert "fish" in sample_dataset.classes
        assert "fish" not in copy.classes


class TestYoloV8DatasetEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_dataset_with_single_class(self, temp_dir):
        """Test dataset with only one class."""
        dataset = YoloV8Dataset.empty(temp_dir, ["only_class"])

        assert len(dataset.classes) == 1
        assert dataset.class_name_to_id == {"only_class": 0}

    def test_dataset_with_many_classes(self, temp_dir):
        """Test dataset with many classes."""
        classes = [f"class_{i}" for i in range(100)]
        dataset = YoloV8Dataset.empty(temp_dir, classes)

        assert len(dataset.classes) == 100
        assert dataset.class_name_to_id["class_0"] == 0
        assert dataset.class_name_to_id["class_99"] == 99

    def test_method_chaining(self, sample_dataset, temp_dir):
        """Test that methods can be chained together."""
        import matplotlib.pyplot as plt

        plt.ioff()

        result = (
            sample_dataset.copy()
            .delete_classes(["fish"])
            .remap_classes({"dog": "canine"})
        )

        assert isinstance(result, YoloV8Dataset)
        assert "canine" in result.classes
        assert "fish" not in result.classes

        plt.close("all")

    def test_operations_on_empty_dataset(self, empty_dataset):
        """Test various operations on empty dataset."""
        # Should not raise errors
        stats = empty_dataset.stats
        assert stats.num_detections == 0

        result = empty_dataset.validate()
        assert result.is_valid

    def test_reproducibility_with_random_seed(self, unbalanced_dataset):
        """Test that operations with random_seed are reproducible."""
        # Balance twice with same seed
        balanced1 = unbalanced_dataset.create_balanced_dataset(random_state=42)
        balanced2 = unbalanced_dataset.create_balanced_dataset(random_state=42)

        # Should get identical results
        assert len(balanced1) == len(balanced2)

        # The actual image selections should be the same
        images1 = set(balanced1.df["image_path"].unique())
        images2 = set(balanced2.df["image_path"].unique())
        assert images1 == images2
