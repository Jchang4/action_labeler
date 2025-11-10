"""Tests for the DetectionDensityFilter and DetectionSizeRankFilter classes.

This module contains comprehensive pytest tests for:
1. DetectionDensityFilter - Filters based on overlap with other detections (IoU)
2. DetectionSizeRankFilter - Filters based on detection size rank (largest/smallest/median)
"""

import numpy as np
import pytest
from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.density import (
    DetectionDensityFilter,
    DetectionSizeRankFilter,
)
from tests.filters.helpers import (
    assert_all_fail,
    assert_all_pass,
    assert_filter_validates_indices,
    count_passing_detections,
    get_failing_indices,
    get_passing_indices,
)


class TestDetectionDensityFilterConstructor:
    """Test cases for DetectionDensityFilter constructor validation."""

    def test_constructor_with_valid_max_overlap_ratio(self):
        """Test that constructor accepts valid max_overlap_ratio values (0.0-1.0)."""
        # Test boundary values
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.0)
        assert filter_obj.max_overlap_ratio == 0.0

        filter_obj = DetectionDensityFilter(max_overlap_ratio=1.0)
        assert filter_obj.max_overlap_ratio == 1.0

        # Test middle value
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.5)
        assert filter_obj.max_overlap_ratio == 0.5

    def test_constructor_with_invalid_max_overlap_ratio_negative(self):
        """Test that negative max_overlap_ratio raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionDensityFilter(max_overlap_ratio=-0.1)

        assert "max_overlap_ratio must be between 0.0 and 1.0" in str(excinfo.value)

    def test_constructor_with_invalid_max_overlap_ratio_above_one(self):
        """Test that max_overlap_ratio > 1.0 raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionDensityFilter(max_overlap_ratio=1.1)

        assert "max_overlap_ratio must be between 0.0 and 1.0" in str(excinfo.value)

    def test_constructor_with_invalid_max_overlap_ratio_large(self):
        """Test that very large max_overlap_ratio values raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionDensityFilter(max_overlap_ratio=5.0)

        assert "max_overlap_ratio must be between 0.0 and 1.0" in str(excinfo.value)

    def test_constructor_default_values(self):
        """Test that constructor uses correct default values."""
        filter_obj = DetectionDensityFilter()
        assert filter_obj.max_overlap_ratio == 0.5
        assert filter_obj.include_crowded is False

    def test_constructor_with_include_crowded_true(self):
        """Test constructor with include_crowded=True."""
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.3, include_crowded=True)
        assert filter_obj.max_overlap_ratio == 0.3
        assert filter_obj.include_crowded is True

    def test_constructor_with_include_crowded_false(self):
        """Test constructor with include_crowded=False."""
        filter_obj = DetectionDensityFilter(
            max_overlap_ratio=0.7, include_crowded=False
        )
        assert filter_obj.max_overlap_ratio == 0.7
        assert filter_obj.include_crowded is False


class TestDetectionDensityFilterWithDefaultSettings:
    """Test DetectionDensityFilter with default settings (exclude overlapping)."""

    def test_default_settings_with_overlapping_detection(self, overlapping_detection):
        """Test default settings exclude overlapping detections.

        overlapping_detection has:
        - Detection 0: [240, 180, 400, 300] - overlaps with 1 (IoU ~0.26)
        - Detection 1: [300, 220, 460, 340] - overlaps with 0 (IoU ~0.26)
        - Detection 2: [500, 50, 600, 150] - isolated, no overlap

        With max_overlap_ratio=0.5, IoU of 0.26 is below threshold,
        so detections 0 and 1 are NOT considered crowded and should pass.
        """
        filter_obj = DetectionDensityFilter()  # max_overlap_ratio=0.5, include_crowded=False
        # All detections should pass since IoU (0.26) < threshold (0.5)
        assert_all_pass(filter_obj, overlapping_detection)

    def test_default_settings_with_single_detection(self, single_detection):
        """Test that single detection always passes (never crowded).

        A single detection has no other detections to overlap with,
        so it's never crowded.
        """
        filter_obj = DetectionDensityFilter()
        assert_all_pass(filter_obj, single_detection)

    def test_default_settings_with_empty_detection(self, empty_detection):
        """Test that empty detection works correctly."""
        filter_obj = DetectionDensityFilter()
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []

    def test_default_settings_with_varying_size_detection(self, varying_size_detection):
        """Test default settings with non-overlapping detections of varying sizes.

        varying_size_detection has 4 non-overlapping detections,
        so all should pass the filter.
        """
        filter_obj = DetectionDensityFilter()
        assert_all_pass(filter_obj, varying_size_detection)


class TestDetectionDensityFilterWithLowThreshold:
    """Test DetectionDensityFilter with low overlap threshold (strict filtering)."""

    def test_low_threshold_excludes_overlapping(self, overlapping_detection):
        """Test low threshold excludes overlapping detections.

        With max_overlap_ratio=0.1, detections 0 and 1 (IoU ~0.26) exceed
        the threshold and should be excluded. Detection 2 (isolated) should pass.
        """
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.1, include_crowded=False)
        # Detections 0 and 1 have IoU of 0.26 > 0.1, so they're crowded
        # Detection 2 is isolated, should pass
        expected_valid_indices = [2]
        assert_filter_validates_indices(
            filter_obj, overlapping_detection, expected_valid_indices
        )

    def test_very_strict_threshold_zero(self, overlapping_detection):
        """Test with threshold of 0.0 (any overlap is too much).

        With max_overlap_ratio=0.0, any overlap causes exclusion.
        Detections 0 and 1 overlap, so only detection 2 should pass.
        """
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.0, include_crowded=False)
        expected_valid_indices = [2]
        assert_filter_validates_indices(
            filter_obj, overlapping_detection, expected_valid_indices
        )

    def test_low_threshold_with_single_detection(self, single_detection):
        """Test that single detection passes even with very strict threshold."""
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.01, include_crowded=False)
        assert_all_pass(filter_obj, single_detection)


class TestDetectionDensityFilterWithHighThreshold:
    """Test DetectionDensityFilter with high overlap threshold (lenient filtering)."""

    def test_high_threshold_includes_all(self, overlapping_detection):
        """Test high threshold includes all detections.

        With max_overlap_ratio=0.8, the IoU of 0.26 is well below threshold,
        so all detections should pass.
        """
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.8, include_crowded=False)
        assert_all_pass(filter_obj, overlapping_detection)

    def test_maximum_threshold_one(self, overlapping_detection):
        """Test with threshold of 1.0 (maximum leniency).

        With max_overlap_ratio=1.0, only perfect overlap (IoU=1.0) would be
        considered crowded, so all detections should pass.
        """
        filter_obj = DetectionDensityFilter(max_overlap_ratio=1.0, include_crowded=False)
        assert_all_pass(filter_obj, overlapping_detection)

    def test_high_threshold_with_varying_size(self, varying_size_detection):
        """Test high threshold with non-overlapping detections."""
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.9, include_crowded=False)
        assert_all_pass(filter_obj, varying_size_detection)


class TestDetectionDensityFilterWithIncludeCrowded:
    """Test DetectionDensityFilter with include_crowded=True (only crowded detections)."""

    def test_include_crowded_with_overlapping_detection(self, overlapping_detection):
        """Test include_crowded=True only includes overlapping detections.

        With max_overlap_ratio=0.1 and include_crowded=True,
        detections 0 and 1 (IoU ~0.26 > 0.1) should pass.
        Detection 2 (isolated) should fail.
        """
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.1, include_crowded=True)
        # Detections 0 and 1 are crowded (IoU > 0.1), so they pass
        # Detection 2 is not crowded, so it fails
        expected_valid_indices = [0, 1]
        assert_filter_validates_indices(
            filter_obj, overlapping_detection, expected_valid_indices
        )

    def test_include_crowded_with_high_threshold(self, overlapping_detection):
        """Test include_crowded=True with high threshold.

        With max_overlap_ratio=0.8, IoU of 0.26 doesn't exceed threshold,
        so no detections are crowded and all should fail.
        """
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.8, include_crowded=True)
        # No detections have IoU > 0.8, so none are crowded
        assert_all_fail(filter_obj, overlapping_detection)

    def test_include_crowded_with_single_detection(self, single_detection):
        """Test include_crowded=True with single detection.

        Single detection is never crowded, so it should fail when
        include_crowded=True.
        """
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.5, include_crowded=True)
        assert_all_fail(filter_obj, single_detection)

    def test_include_crowded_with_non_overlapping(self, varying_size_detection):
        """Test include_crowded=True with non-overlapping detections.

        Non-overlapping detections are not crowded, so all should fail.
        """
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.5, include_crowded=True)
        assert_all_fail(filter_obj, varying_size_detection)

    def test_include_crowded_with_empty_detection(self, empty_detection):
        """Test include_crowded=True with empty detection."""
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.5, include_crowded=True)
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestDetectionDensityFilterIoUCalculation:
    """Test IoU calculation accuracy with custom detections."""

    def test_iou_calculation_perfect_overlap(self, sample_image):
        """Test IoU calculation with identical boxes (IoU = 1.0)."""
        # Create two identical boxes
        xyxy = np.array([[100, 100, 200, 200], [100, 100, 200, 200]])
        class_id = np.array([0, 0])
        segmentation_points = [
            [0.156, 0.208, 0.312, 0.208, 0.312, 0.417, 0.156, 0.417],
            [0.156, 0.208, 0.312, 0.208, 0.312, 0.417, 0.156, 0.417],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # With threshold 0.99, IoU of 1.0 exceeds it, so detections are crowded
        filter_obj = DetectionDensityFilter(
            max_overlap_ratio=0.99, include_crowded=False
        )
        assert_all_fail(filter_obj, detection)

        # With include_crowded=True, both should pass
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.99, include_crowded=True)
        assert_all_pass(filter_obj, detection)

    def test_iou_calculation_half_overlap(self, sample_image):
        """Test IoU calculation with 50% overlap.

        Two boxes where one is shifted by half its width/height.
        IoU should be approximately 0.33.
        """
        # Box 1: [0, 0, 100, 100]
        # Box 2: [50, 50, 150, 150] - shifted by half
        # Intersection: [50, 50, 100, 100] = 50x50 = 2500
        # Union: 100*100 + 100*100 - 2500 = 17500
        # IoU = 2500 / 17500 = 0.142857 (approximately 1/7)
        xyxy = np.array([[0, 0, 100, 100], [50, 50, 150, 150]])
        class_id = np.array([0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.078, 0.104, 0.234, 0.104, 0.234, 0.312, 0.078, 0.312],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Threshold 0.1: IoU ~0.14 > 0.1, so crowded, should be excluded
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.1, include_crowded=False)
        assert_all_fail(filter_obj, detection)

        # Threshold 0.2: IoU ~0.14 < 0.2, so not crowded, should pass
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.2, include_crowded=False)
        assert_all_pass(filter_obj, detection)

    def test_iou_calculation_no_overlap(self, sample_image):
        """Test IoU calculation with non-overlapping boxes (IoU = 0.0)."""
        xyxy = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
        class_id = np.array([0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.312, 0.417, 0.469, 0.417, 0.469, 0.625, 0.312, 0.625],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # IoU is 0.0, so not crowded regardless of threshold
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.0, include_crowded=False)
        assert_all_pass(filter_obj, detection)

    def test_iou_calculation_touching_boxes(self, sample_image):
        """Test IoU calculation with boxes that touch but don't overlap."""
        # Boxes share an edge but have no area overlap
        xyxy = np.array([[0, 0, 100, 100], [100, 0, 200, 100]])
        class_id = np.array([0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.156, 0.0, 0.312, 0.0, 0.312, 0.208, 0.156, 0.208],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Touching boxes have IoU of 0.0
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.0, include_crowded=False)
        assert_all_pass(filter_obj, detection)

    def test_iou_calculation_zero_area_boxes(self, sample_image):
        """Test IoU calculation with zero-area boxes (union_area = 0).

        This tests the edge case where both boxes have zero area,
        resulting in union_area = 0. The IoU should return 0.0.
        This covers the defensive code path in _calculate_iou.
        """
        # Create two boxes with zero area at the same point (complete overlap but zero area)
        # This creates: intersection_area=0, box1_area=0, box2_area=0, union_area=0
        xyxy = np.array([[100, 100, 100, 100], [100, 100, 100, 100]])
        class_id = np.array([0, 0])
        segmentation_points = [
            [0.156, 0.208, 0.156, 0.208, 0.156, 0.208, 0.156, 0.208],
            [0.156, 0.208, 0.156, 0.208, 0.156, 0.208, 0.156, 0.208],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Zero area boxes have IoU of 0.0 (not crowded)
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.0, include_crowded=False)
        assert_all_pass(filter_obj, detection)


class TestDetectionDensityFilterBoundaryConditions:
    """Test boundary conditions for DetectionDensityFilter."""

    def test_detection_exactly_at_threshold(self, sample_image):
        """Test detection with IoU exactly at the threshold.

        When IoU equals threshold, detection should NOT be considered crowded
        (using > not >=).
        """
        # Create boxes with specific IoU
        # Box 1: [0, 0, 100, 100]
        # Box 2: [50, 50, 150, 150]
        # IoU = 2500 / 17500 ≈ 0.142857
        xyxy = np.array([[0, 0, 100, 100], [50, 50, 150, 150]])
        class_id = np.array([0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.078, 0.104, 0.234, 0.104, 0.234, 0.312, 0.078, 0.312],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Set threshold to approximately the IoU value
        iou_value = 2500 / 17500
        filter_obj = DetectionDensityFilter(
            max_overlap_ratio=iou_value, include_crowded=False
        )
        # IoU == threshold, so NOT crowded (uses >), should pass
        assert_all_pass(filter_obj, detection)

        # Just below threshold - should pass
        filter_obj = DetectionDensityFilter(
            max_overlap_ratio=iou_value - 0.001, include_crowded=False
        )
        assert_all_fail(filter_obj, detection)

        # Just above threshold - should pass
        filter_obj = DetectionDensityFilter(
            max_overlap_ratio=iou_value + 0.001, include_crowded=False
        )
        assert_all_pass(filter_obj, detection)

    def test_multiple_detections_max_iou_selection(self, sample_image):
        """Test that filter uses maximum IoU when detection overlaps with multiple others.

        If a detection overlaps with multiple other detections, it should use
        the maximum IoU to determine if it's crowded.
        """
        # Create three boxes where box 0 overlaps with both box 1 and box 2
        # but has different IoU with each
        xyxy = np.array(
            [
                [100, 100, 200, 200],  # Box 0
                [150, 150, 250, 250],  # Box 1 - overlaps with 0
                [120, 120, 180, 180],  # Box 2 - heavily overlaps with 0
            ]
        )
        class_id = np.array([0, 0, 0])
        segmentation_points = [
            [0.156, 0.208, 0.312, 0.208, 0.312, 0.417, 0.156, 0.417],
            [0.234, 0.312, 0.391, 0.312, 0.391, 0.521, 0.234, 0.521],
            [0.188, 0.250, 0.281, 0.250, 0.281, 0.375, 0.188, 0.375],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Box 2 is contained within box 0, creating high IoU
        # Box 0 should be flagged as crowded due to overlap with box 2
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.3, include_crowded=False)
        passing = get_passing_indices(filter_obj, detection)
        # At least some should be filtered out due to high overlap
        assert len(passing) < 3


class TestDetectionDensityFilterHelperFunctions:
    """Test DetectionDensityFilter using helper functions."""

    def test_count_passing_detections(self, overlapping_detection):
        """Test counting how many detections pass the filter."""
        # With strict threshold, only isolated detection passes
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.1, include_crowded=False)
        count = count_passing_detections(filter_obj, overlapping_detection)
        assert count == 1

        # With lenient threshold, all pass
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.8, include_crowded=False)
        count = count_passing_detections(filter_obj, overlapping_detection)
        assert count == 3

    def test_get_passing_indices(self, overlapping_detection):
        """Test getting list of indices that pass the filter."""
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.1, include_crowded=False)
        passing = get_passing_indices(filter_obj, overlapping_detection)
        assert passing == [2]

    def test_get_failing_indices(self, overlapping_detection):
        """Test getting list of indices that fail the filter."""
        filter_obj = DetectionDensityFilter(max_overlap_ratio=0.1, include_crowded=False)
        failing = get_failing_indices(filter_obj, overlapping_detection)
        assert set(failing) == {0, 1}


# ============================================================================
# DetectionSizeRankFilter Tests
# ============================================================================


class TestDetectionSizeRankFilterConstructor:
    """Test cases for DetectionSizeRankFilter constructor validation."""

    def test_constructor_with_valid_largest_rank(self):
        """Test constructor with valid 'largest' rank."""
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        assert filter_obj.rank == "largest"
        assert filter_obj.top_n == 1

    def test_constructor_with_valid_smallest_rank(self):
        """Test constructor with valid 'smallest' rank."""
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=1)
        assert filter_obj.rank == "smallest"
        assert filter_obj.bottom_n == 1

    def test_constructor_with_valid_median_rank(self):
        """Test constructor with valid 'median' rank."""
        filter_obj = DetectionSizeRankFilter(rank="median")
        assert filter_obj.rank == "median"

    def test_constructor_with_invalid_rank(self):
        """Test that invalid rank raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionSizeRankFilter(rank="middle")

        assert "rank must be one of" in str(excinfo.value)
        assert "'middle'" in str(excinfo.value)

    def test_constructor_with_invalid_rank_empty_string(self):
        """Test that empty string rank raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionSizeRankFilter(rank="")

        assert "rank must be one of" in str(excinfo.value)

    def test_constructor_with_invalid_top_n_zero(self):
        """Test that top_n=0 raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionSizeRankFilter(rank="largest", top_n=0)

        assert "top_n must be at least 1" in str(excinfo.value)

    def test_constructor_with_invalid_top_n_negative(self):
        """Test that negative top_n raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionSizeRankFilter(rank="largest", top_n=-1)

        assert "top_n must be at least 1" in str(excinfo.value)

    def test_constructor_with_invalid_bottom_n_zero(self):
        """Test that bottom_n=0 raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionSizeRankFilter(rank="smallest", bottom_n=0)

        assert "bottom_n must be at least 1" in str(excinfo.value)

    def test_constructor_with_invalid_bottom_n_negative(self):
        """Test that negative bottom_n raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            DetectionSizeRankFilter(rank="smallest", bottom_n=-5)

        assert "bottom_n must be at least 1" in str(excinfo.value)

    def test_constructor_default_values(self):
        """Test that constructor uses correct default values."""
        filter_obj = DetectionSizeRankFilter()
        assert filter_obj.rank == "largest"
        assert filter_obj.top_n == 1
        assert filter_obj.bottom_n == 1

    def test_constructor_with_large_top_n(self):
        """Test constructor with large top_n value."""
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=100)
        assert filter_obj.top_n == 100

    def test_constructor_with_large_bottom_n(self):
        """Test constructor with large bottom_n value."""
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=50)
        assert filter_obj.bottom_n == 50


class TestDetectionSizeRankFilterLargest:
    """Test DetectionSizeRankFilter with rank='largest'."""

    def test_largest_top_n_one(self, varying_size_detection):
        """Test rank='largest' with top_n=1 - only largest detection passes.

        varying_size_detection has (ordered by area):
        - Detection 0: Large (200x150 = 30,000 px²) - should pass
        - Detection 1: Medium (100x80 = 8,000 px²)
        - Detection 2: Small (50x40 = 2,000 px²)
        - Detection 3: Tiny (20x20 = 400 px²)
        """
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        expected_valid_indices = [0]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_largest_top_n_two(self, varying_size_detection):
        """Test rank='largest' with top_n=2 - two largest detections pass.

        Detections 0 and 1 should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=2)
        expected_valid_indices = [0, 1]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_largest_top_n_three(self, varying_size_detection):
        """Test rank='largest' with top_n=3 - three largest detections pass.

        Detections 0, 1, and 2 should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=3)
        expected_valid_indices = [0, 1, 2]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_largest_top_n_all(self, varying_size_detection):
        """Test rank='largest' with top_n equal to total detections.

        All detections should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=4)
        assert_all_pass(filter_obj, varying_size_detection)

    def test_largest_top_n_exceeds_total(self, varying_size_detection):
        """Test rank='largest' with top_n exceeding total detections.

        All detections should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=10)
        assert_all_pass(filter_obj, varying_size_detection)

    def test_largest_with_single_detection(self, single_detection):
        """Test rank='largest' with single detection.

        Single detection is the largest, so it should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        assert_all_pass(filter_obj, single_detection)

    def test_largest_with_empty_detection(self, empty_detection):
        """Test rank='largest' with empty detection."""
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestDetectionSizeRankFilterSmallest:
    """Test DetectionSizeRankFilter with rank='smallest'."""

    def test_smallest_bottom_n_one(self, varying_size_detection):
        """Test rank='smallest' with bottom_n=1 - only smallest detection passes.

        varying_size_detection has (ordered by area):
        - Detection 0: Large (200x150 = 30,000 px²)
        - Detection 1: Medium (100x80 = 8,000 px²)
        - Detection 2: Small (50x40 = 2,000 px²)
        - Detection 3: Tiny (20x20 = 400 px²) - should pass
        """
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=1)
        expected_valid_indices = [3]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_smallest_bottom_n_two(self, varying_size_detection):
        """Test rank='smallest' with bottom_n=2 - two smallest detections pass.

        Detections 2 and 3 should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=2)
        expected_valid_indices = [2, 3]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_smallest_bottom_n_three(self, varying_size_detection):
        """Test rank='smallest' with bottom_n=3 - three smallest detections pass.

        Detections 1, 2, and 3 should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=3)
        expected_valid_indices = [1, 2, 3]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_smallest_bottom_n_all(self, varying_size_detection):
        """Test rank='smallest' with bottom_n equal to total detections.

        All detections should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=4)
        assert_all_pass(filter_obj, varying_size_detection)

    def test_smallest_bottom_n_exceeds_total(self, varying_size_detection):
        """Test rank='smallest' with bottom_n exceeding total detections.

        All detections should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=10)
        assert_all_pass(filter_obj, varying_size_detection)

    def test_smallest_with_single_detection(self, single_detection):
        """Test rank='smallest' with single detection.

        Single detection is the smallest, so it should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=1)
        assert_all_pass(filter_obj, single_detection)

    def test_smallest_with_empty_detection(self, empty_detection):
        """Test rank='smallest' with empty detection."""
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=1)
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestDetectionSizeRankFilterMedian:
    """Test DetectionSizeRankFilter with rank='median'."""

    def test_median_with_odd_number_detections(self, sample_image):
        """Test rank='median' with odd number of detections (3 detections).

        With 3 detections, median is at index 1 (middle element).
        """
        # Create 3 detections of different sizes
        xyxy = np.array(
            [
                [0, 0, 100, 100],  # 10,000 px² - largest
                [200, 200, 250, 250],  # 2,500 px² - median (middle)
                [300, 300, 310, 310],  # 100 px² - smallest
            ]
        )
        class_id = np.array([0, 0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.312, 0.417, 0.391, 0.417, 0.391, 0.521, 0.312, 0.521],
            [0.469, 0.625, 0.484, 0.625, 0.484, 0.646, 0.469, 0.646],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = DetectionSizeRankFilter(rank="median")
        # Median index is 3 // 2 = 1, which corresponds to detection 1
        expected_valid_indices = [1]
        assert_filter_validates_indices(filter_obj, detection, expected_valid_indices)

    def test_median_with_even_number_detections(self, varying_size_detection):
        """Test rank='median' with even number of detections (4 detections).

        With 4 detections, median is at index 2 (len // 2 = 4 // 2 = 2).
        varying_size_detection sorted by size: [0, 1, 2, 3] (largest to smallest)
        Median index 2 corresponds to detection 2.
        """
        filter_obj = DetectionSizeRankFilter(rank="median")
        expected_valid_indices = [2]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_median_with_single_detection(self, single_detection):
        """Test rank='median' with single detection.

        Single detection is the median, so it should pass.
        """
        filter_obj = DetectionSizeRankFilter(rank="median")
        assert_all_pass(filter_obj, single_detection)

    def test_median_with_two_detections(self, sample_image):
        """Test rank='median' with two detections.

        With 2 detections, median is at index 1 (len // 2 = 2 // 2 = 1).
        """
        xyxy = np.array([[0, 0, 100, 100], [200, 200, 250, 250]])
        class_id = np.array([0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.312, 0.417, 0.391, 0.417, 0.391, 0.521, 0.312, 0.521],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = DetectionSizeRankFilter(rank="median")
        # Median index is 2 // 2 = 1, which is the smaller detection
        expected_valid_indices = [1]
        assert_filter_validates_indices(filter_obj, detection, expected_valid_indices)

    def test_median_with_empty_detection(self, empty_detection):
        """Test rank='median' with empty detection."""
        filter_obj = DetectionSizeRankFilter(rank="median")
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestDetectionSizeRankFilterSameSizeDetections:
    """Test edge cases with all same size detections."""

    def test_largest_with_all_same_size(self, sample_image):
        """Test rank='largest' when all detections have the same size.

        With top_n=1 and all same size, only the first in sorted order should pass.
        """
        # Create 3 identical-sized detections (100x100 each)
        xyxy = np.array(
            [
                [0, 0, 100, 100],
                [200, 0, 300, 100],
                [400, 0, 500, 100],
            ]
        )
        class_id = np.array([0, 0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.312, 0.0, 0.469, 0.0, 0.469, 0.208, 0.312, 0.208],
            [0.625, 0.0, 0.781, 0.0, 0.781, 0.208, 0.625, 0.208],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # With same size, top_n=1 should select first one
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        count = count_passing_detections(filter_obj, detection)
        assert count == 1

        # With top_n=2, should select first two
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=2)
        count = count_passing_detections(filter_obj, detection)
        assert count == 2

    def test_smallest_with_all_same_size(self, sample_image):
        """Test rank='smallest' when all detections have the same size.

        With bottom_n=1 and all same size, only the last in sorted order should pass.
        """
        # Create 3 identical-sized detections (100x100 each)
        xyxy = np.array(
            [
                [0, 0, 100, 100],
                [200, 0, 300, 100],
                [400, 0, 500, 100],
            ]
        )
        class_id = np.array([0, 0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.312, 0.0, 0.469, 0.0, 0.469, 0.208, 0.312, 0.208],
            [0.625, 0.0, 0.781, 0.0, 0.781, 0.208, 0.625, 0.208],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # With same size, bottom_n=1 should select last one
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=1)
        count = count_passing_detections(filter_obj, detection)
        assert count == 1

        # With bottom_n=3, all should pass
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=3)
        assert_all_pass(filter_obj, detection)

    def test_median_with_all_same_size(self, sample_image):
        """Test rank='median' when all detections have the same size.

        Median index should select middle detection.
        """
        # Create 3 identical-sized detections (100x100 each)
        xyxy = np.array(
            [
                [0, 0, 100, 100],
                [200, 0, 300, 100],
                [400, 0, 500, 100],
            ]
        )
        class_id = np.array([0, 0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.312, 0.0, 0.469, 0.0, 0.469, 0.208, 0.312, 0.208],
            [0.625, 0.0, 0.781, 0.0, 0.781, 0.208, 0.625, 0.208],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = DetectionSizeRankFilter(rank="median")
        # Median index is 3 // 2 = 1
        count = count_passing_detections(filter_obj, detection)
        assert count == 1


class TestDetectionSizeRankFilterHelperFunctions:
    """Test DetectionSizeRankFilter using helper functions."""

    def test_count_passing_detections_largest(self, varying_size_detection):
        """Test counting passing detections for largest rank."""
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=2)
        count = count_passing_detections(filter_obj, varying_size_detection)
        assert count == 2

    def test_count_passing_detections_smallest(self, varying_size_detection):
        """Test counting passing detections for smallest rank."""
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=2)
        count = count_passing_detections(filter_obj, varying_size_detection)
        assert count == 2

    def test_get_passing_indices_largest(self, varying_size_detection):
        """Test getting passing indices for largest rank."""
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        passing = get_passing_indices(filter_obj, varying_size_detection)
        assert passing == [0]

    def test_get_passing_indices_smallest(self, varying_size_detection):
        """Test getting passing indices for smallest rank."""
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=1)
        passing = get_passing_indices(filter_obj, varying_size_detection)
        assert passing == [3]

    def test_get_failing_indices_largest(self, varying_size_detection):
        """Test getting failing indices for largest rank."""
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        failing = get_failing_indices(filter_obj, varying_size_detection)
        assert set(failing) == {1, 2, 3}

    def test_get_failing_indices_smallest(self, varying_size_detection):
        """Test getting failing indices for smallest rank."""
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=1)
        failing = get_failing_indices(filter_obj, varying_size_detection)
        assert set(failing) == {0, 1, 2}


class TestDetectionSizeRankFilterAreaCalculation:
    """Test area calculation for DetectionSizeRankFilter."""

    def test_area_calculation_accuracy(self, sample_image):
        """Test that area is calculated correctly for size ranking.

        Create detections with known areas and verify correct ranking.
        """
        # Create detections with specific areas
        # Detection 0: 100x100 = 10,000 px²
        # Detection 1: 50x200 = 10,000 px² (same area, different shape)
        # Detection 2: 200x50 = 10,000 px² (same area, different shape)
        # Detection 3: 150x150 = 22,500 px² (largest)
        xyxy = np.array(
            [
                [0, 0, 100, 100],  # 10,000
                [200, 0, 250, 200],  # 10,000
                [300, 0, 500, 50],  # 10,000
                [0, 200, 150, 350],  # 22,500
            ]
        )
        class_id = np.array([0, 0, 0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208],
            [0.312, 0.0, 0.391, 0.0, 0.391, 0.417, 0.312, 0.417],
            [0.469, 0.0, 0.781, 0.0, 0.781, 0.104, 0.469, 0.104],
            [0.0, 0.417, 0.234, 0.417, 0.234, 0.729, 0.0, 0.729],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Detection 3 has the largest area
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        expected_valid_indices = [3]
        assert_filter_validates_indices(filter_obj, detection, expected_valid_indices)

        # Detections 0, 1, 2 are smallest (all same area)
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=3)
        passing = get_passing_indices(filter_obj, detection)
        assert set(passing) == {0, 1, 2}

    def test_area_calculation_with_very_small_detections(self, sample_image):
        """Test area calculation with very small detection boxes."""
        # Create very small detections (1x1, 2x2, 3x3 pixels)
        xyxy = np.array(
            [
                [0, 0, 1, 1],  # 1 px²
                [10, 10, 12, 12],  # 4 px²
                [20, 20, 23, 23],  # 9 px²
            ]
        )
        class_id = np.array([0, 0, 0])
        segmentation_points = [
            [0.0, 0.0, 0.002, 0.0, 0.002, 0.002, 0.0, 0.002],
            [0.016, 0.021, 0.019, 0.021, 0.019, 0.025, 0.016, 0.025],
            [0.031, 0.042, 0.036, 0.042, 0.036, 0.048, 0.031, 0.048],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Detection 2 is largest (9 px²)
        filter_obj = DetectionSizeRankFilter(rank="largest", top_n=1)
        expected_valid_indices = [2]
        assert_filter_validates_indices(filter_obj, detection, expected_valid_indices)

        # Detection 0 is smallest (1 px²)
        filter_obj = DetectionSizeRankFilter(rank="smallest", bottom_n=1)
        expected_valid_indices = [0]
        assert_filter_validates_indices(filter_obj, detection, expected_valid_indices)
