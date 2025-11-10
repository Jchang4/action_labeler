"""Tests for the ratio filter classes.

This module contains comprehensive pytest tests for the ratio-based filters:
- SmallDetectionsFilter: Filters out detections that are too small relative to image size
- MinDetectionSizeFilter: Approves detections that meet minimum pixel dimensions

The varying_size_detection fixture (640x480 = 307,200 px²) contains:
- Detection 0: 200x150 = 30,000 px² (9.77% of image)
- Detection 1: 100x80 = 8,000 px² (2.6% of image)
- Detection 2: 50x40 = 2,000 px² (0.65% of image)
- Detection 3: 20x20 = 400 px² (0.13% of image)
"""

import numpy as np
import pytest

from action_labeler.detections.detection import Detection
from action_labeler.filters.ratio import MinDetectionSizeFilter, SmallDetectionsFilter
from tests.filters.helpers import (
    assert_all_fail,
    assert_all_pass,
    assert_filter_validates_indices,
    count_passing_detections,
    get_failing_indices,
    get_passing_indices,
)


class TestSmallDetectionsFilterConstructor:
    """Test cases for SmallDetectionsFilter constructor."""

    def test_constructor_default_min_area(self):
        """Test that constructor uses default min_area of 0.05 (5%).

        The default threshold should be 5% of the image area.
        """
        filter_obj = SmallDetectionsFilter()
        assert filter_obj.min_area == 0.05

    def test_constructor_custom_min_area(self):
        """Test that constructor accepts custom min_area values."""
        filter_obj = SmallDetectionsFilter(min_area=0.10)
        assert filter_obj.min_area == 0.10

        filter_obj = SmallDetectionsFilter(min_area=0.01)
        assert filter_obj.min_area == 0.01

    def test_constructor_accepts_zero(self):
        """Test that constructor accepts min_area of 0.

        A min_area of 0 means all detections should pass (no minimum).
        """
        filter_obj = SmallDetectionsFilter(min_area=0.0)
        assert filter_obj.min_area == 0.0


class TestSmallDetectionsFilterDefaultThreshold:
    """Test SmallDetectionsFilter with default min_area=0.05 (5%)."""

    def test_default_threshold_filters_correctly(self, varying_size_detection):
        """Test that default 5% threshold filters detections correctly.

        With min_area=0.05 and image 640x480 (307,200 px²):
        - Detection 0: 30,000 px² = 9.77% (should pass)
        - Detection 1: 8,000 px² = 2.6% (should fail)
        - Detection 2: 2,000 px² = 0.65% (should fail)
        - Detection 3: 400 px² = 0.13% (should fail)
        """
        filter_obj = SmallDetectionsFilter()  # default min_area=0.05

        # Only detection 0 should pass (9.77% > 5%)
        expected_valid_indices = [0]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_default_threshold_count_passing(self, varying_size_detection):
        """Test that exactly 1 detection passes with default threshold.

        Using helper function to verify count.
        """
        filter_obj = SmallDetectionsFilter()
        count = count_passing_detections(filter_obj, varying_size_detection)
        assert count == 1

    def test_default_threshold_get_passing_indices(self, varying_size_detection):
        """Test getting list of passing indices with default threshold."""
        filter_obj = SmallDetectionsFilter()
        passing = get_passing_indices(filter_obj, varying_size_detection)
        assert passing == [0]

    def test_default_threshold_get_failing_indices(self, varying_size_detection):
        """Test getting list of failing indices with default threshold."""
        filter_obj = SmallDetectionsFilter()
        failing = get_failing_indices(filter_obj, varying_size_detection)
        assert failing == [1, 2, 3]


class TestSmallDetectionsFilterPermissiveThreshold:
    """Test SmallDetectionsFilter with min_area=0.01 (1%) - more permissive."""

    def test_permissive_threshold_filters_correctly(self, varying_size_detection):
        """Test that 1% threshold allows more detections to pass.

        With min_area=0.01 and image 640x480 (307,200 px²):
        - Detection 0: 30,000 px² = 9.77% (should pass)
        - Detection 1: 8,000 px² = 2.6% (should pass)
        - Detection 2: 2,000 px² = 0.65% (should fail)
        - Detection 3: 400 px² = 0.13% (should fail)
        """
        filter_obj = SmallDetectionsFilter(min_area=0.01)

        # Detections 0 and 1 should pass (9.77% and 2.6% > 1%)
        expected_valid_indices = [0, 1]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_permissive_threshold_count_passing(self, varying_size_detection):
        """Test that exactly 2 detections pass with 1% threshold."""
        filter_obj = SmallDetectionsFilter(min_area=0.01)
        count = count_passing_detections(filter_obj, varying_size_detection)
        assert count == 2

    def test_very_permissive_threshold(self, varying_size_detection):
        """Test that 0.001% threshold allows even more detections.

        With min_area=0.001 and image 640x480 (307,200 px²):
        - Detection 0: 9.77% (should pass)
        - Detection 1: 2.6% (should pass)
        - Detection 2: 0.65% (should pass)
        - Detection 3: 0.13% (should pass)
        """
        filter_obj = SmallDetectionsFilter(min_area=0.001)

        # All detections should pass (all > 0.1%)
        assert_all_pass(filter_obj, varying_size_detection)


class TestSmallDetectionsFilterRestrictiveThreshold:
    """Test SmallDetectionsFilter with min_area=0.10 (10%) - very restrictive."""

    def test_restrictive_threshold_filters_correctly(self, varying_size_detection):
        """Test that 10% threshold is very restrictive.

        With min_area=0.10 and image 640x480 (307,200 px²):
        - Detection 0: 30,000 px² = 9.77% (should fail)
        - Detection 1: 8,000 px² = 2.6% (should fail)
        - Detection 2: 2,000 px² = 0.65% (should fail)
        - Detection 3: 400 px² = 0.13% (should fail)

        All detections fail because none reach 10% of image area.
        """
        filter_obj = SmallDetectionsFilter(min_area=0.10)

        # No detections should pass (largest is 9.77% < 10%)
        assert_all_fail(filter_obj, varying_size_detection)

    def test_restrictive_threshold_count_passing(self, varying_size_detection):
        """Test that zero detections pass with 10% threshold."""
        filter_obj = SmallDetectionsFilter(min_area=0.10)
        count = count_passing_detections(filter_obj, varying_size_detection)
        assert count == 0


class TestSmallDetectionsFilterBoundaryConditions:
    """Test SmallDetectionsFilter boundary conditions."""

    def test_boundary_detection_exactly_at_threshold(self, sample_image):
        """Test detection with area exactly at the threshold.

        Create a detection with exactly 5% of image area and verify it passes
        with min_area=0.05.

        Image: 640x480 = 307,200 px²
        5% = 15,360 px²
        Box: 128x120 = 15,360 px² (exactly 5%)
        """
        # Create detection with area exactly at 5% threshold
        # 640x480 = 307,200 px², 5% = 15,360 px²
        # Use 128x120 box = 15,360 px²
        xyxy = np.array([[100, 100, 228, 220]])  # 128x120 = 15,360 px²
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.356, 0.208, 0.356, 0.458, 0.156, 0.458]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = SmallDetectionsFilter(min_area=0.05)

        # Detection should pass (area >= threshold)
        assert_all_pass(filter_obj, detection)

    def test_boundary_detection_just_below_threshold(self, sample_image):
        """Test detection with area just below the threshold.

        Create a detection with slightly less than 5% of image area and verify
        it fails with min_area=0.05.

        Image: 640x480 = 307,200 px²
        5% = 15,360 px²
        Box: 127x120 = 15,240 px² (4.96% < 5%)
        """
        # Create detection with area just below 5% threshold
        # Use 127x120 box = 15,240 px² (4.96%)
        xyxy = np.array([[100, 100, 227, 220]])  # 127x120 = 15,240 px²
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.355, 0.208, 0.355, 0.458, 0.156, 0.458]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = SmallDetectionsFilter(min_area=0.05)

        # Detection should fail (area < threshold)
        assert_all_fail(filter_obj, detection)

    def test_boundary_detection_just_above_threshold(self, sample_image):
        """Test detection with area just above the threshold.

        Create a detection with slightly more than 5% of image area and verify
        it passes with min_area=0.05.

        Image: 640x480 = 307,200 px²
        5% = 15,360 px²
        Box: 129x120 = 15,480 px² (5.04% > 5%)
        """
        # Create detection with area just above 5% threshold
        # Use 129x120 box = 15,480 px² (5.04%)
        xyxy = np.array([[100, 100, 229, 220]])  # 129x120 = 15,480 px²
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.358, 0.208, 0.358, 0.458, 0.156, 0.458]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = SmallDetectionsFilter(min_area=0.05)

        # Detection should pass (area > threshold)
        assert_all_pass(filter_obj, detection)


class TestSmallDetectionsFilterEmptyAndSingleDetection:
    """Test SmallDetectionsFilter with empty and single detection fixtures."""

    def test_empty_detection(self, empty_detection):
        """Test that empty detection works correctly.

        With no detections, the filter should not raise errors.
        """
        filter_obj = SmallDetectionsFilter()
        assert len(empty_detection.xyxy) == 0

        # Verify this doesn't raise an error
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []

    def test_single_detection_passes(self, single_detection):
        """Test single detection with default threshold.

        single_detection has 128x96 box = 12,288 px² in 640x480 image
        12,288 / 307,200 = 4% (should fail default 5% threshold)
        """
        filter_obj = SmallDetectionsFilter()  # default min_area=0.05

        # Single detection is 4% < 5%, should fail
        assert_all_fail(filter_obj, single_detection)

    def test_single_detection_with_lower_threshold(self, single_detection):
        """Test single detection with lower threshold.

        single_detection is 4% of image, should pass with 3% threshold.
        """
        filter_obj = SmallDetectionsFilter(min_area=0.03)

        # Single detection is 4% > 3%, should pass
        assert_all_pass(filter_obj, single_detection)


class TestSmallDetectionsFilterDifferentImageSizes:
    """Test SmallDetectionsFilter with different image sizes."""

    def test_small_image_different_ratios(self, small_image):
        """Test that same pixel box has different ratio on different image size.

        small_image is 100x100 = 10,000 px²
        A 50x40 box = 2,000 px² is 20% of small_image
        but the same box is 0.65% of sample_image (640x480)
        """
        # Create detection with 50x40 box on small 100x100 image
        # 2,000 px² / 10,000 px² = 20%
        xyxy = np.array([[25, 30, 75, 70]])  # 50x40 = 2,000 px²
        class_id = np.array([0])
        segmentation_points = [[0.25, 0.30, 0.75, 0.30, 0.75, 0.70, 0.25, 0.70]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=small_image,
        )

        # With 5% threshold, should pass (20% > 5%)
        filter_obj = SmallDetectionsFilter(min_area=0.05)
        assert_all_pass(filter_obj, detection)

        # With 25% threshold, should fail (20% < 25%)
        filter_obj = SmallDetectionsFilter(min_area=0.25)
        assert_all_fail(filter_obj, detection)

    def test_varying_size_on_small_image(self, small_image):
        """Test varying_size_detection pattern on small image.

        Create similar varying sizes but on 100x100 image to test
        how relative sizing changes behavior.
        """
        # On 100x100 image (10,000 px²):
        # - 60x50 = 3,000 px² (30%)
        # - 40x30 = 1,200 px² (12%)
        # - 20x15 = 300 px² (3%)
        # - 10x10 = 100 px² (1%)
        xyxy = np.array(
            [
                [20, 25, 80, 75],   # 60x50 = 30%
                [10, 10, 50, 40],   # 40x30 = 12%
                [70, 70, 90, 85],   # 20x15 = 3%
                [5, 5, 15, 15],     # 10x10 = 1%
            ]
        )
        class_id = np.array([0, 0, 0, 0])
        segmentation_points = [
            [0.20, 0.25, 0.80, 0.25, 0.80, 0.75, 0.20, 0.75],
            [0.10, 0.10, 0.50, 0.10, 0.50, 0.40, 0.10, 0.40],
            [0.70, 0.70, 0.90, 0.70, 0.90, 0.85, 0.70, 0.85],
            [0.05, 0.05, 0.15, 0.05, 0.15, 0.15, 0.05, 0.15],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=small_image,
        )

        # With 5% threshold, detections 0, 1, 2 should fail (30%, 12%, 3% > 5%)
        # detection 3 should fail (1% < 5%)
        filter_obj = SmallDetectionsFilter(min_area=0.05)
        expected_valid_indices = [0, 1]  # 30% and 12% pass
        assert_filter_validates_indices(filter_obj, detection, expected_valid_indices)


class TestMinDetectionSizeFilterConstructor:
    """Test cases for MinDetectionSizeFilter constructor."""

    def test_constructor_default_min_pixels(self):
        """Test that constructor uses default min_pixels of 300.

        The default threshold requires both width and height >= 300 pixels.
        """
        filter_obj = MinDetectionSizeFilter()
        assert filter_obj.min_pixels == 300

    def test_constructor_custom_min_pixels(self):
        """Test that constructor accepts custom min_pixels values."""
        filter_obj = MinDetectionSizeFilter(min_pixels=100)
        assert filter_obj.min_pixels == 100

        filter_obj = MinDetectionSizeFilter(min_pixels=50)
        assert filter_obj.min_pixels == 50


class TestMinDetectionSizeFilterDefaultThreshold:
    """Test MinDetectionSizeFilter with default min_pixels=300."""

    def test_default_threshold_all_fail(self, varying_size_detection):
        """Test that default 300px threshold filters out all detections.

        With min_pixels=300:
        - Detection 0: 200x150 pixels (both < 300, should fail)
        - Detection 1: 100x80 pixels (both < 300, should fail)
        - Detection 2: 50x40 pixels (both < 300, should fail)
        - Detection 3: 20x20 pixels (both < 300, should fail)

        All detections fail because none have both width >= 300 AND height >= 300.
        """
        filter_obj = MinDetectionSizeFilter()  # default min_pixels=300

        # All detections should fail
        assert_all_fail(filter_obj, varying_size_detection)

    def test_default_threshold_count_passing(self, varying_size_detection):
        """Test that zero detections pass with default threshold."""
        filter_obj = MinDetectionSizeFilter()
        count = count_passing_detections(filter_obj, varying_size_detection)
        assert count == 0


class TestMinDetectionSizeFilterCustomThresholds:
    """Test MinDetectionSizeFilter with various custom thresholds."""

    def test_min_pixels_50_filters_correctly(self, varying_size_detection):
        """Test that min_pixels=50 allows appropriate detections.

        With min_pixels=50:
        - Detection 0: 200x150 pixels (both >= 50, should pass)
        - Detection 1: 100x80 pixels (both >= 50, should pass)
        - Detection 2: 50x40 pixels (width=50 >=, height=40 <, should fail)
        - Detection 3: 20x20 pixels (both < 50, should fail)
        """
        filter_obj = MinDetectionSizeFilter(min_pixels=50)

        # Detections 0 and 1 should pass
        expected_valid_indices = [0, 1]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_min_pixels_100_filters_correctly(self, varying_size_detection):
        """Test that min_pixels=100 filters more strictly.

        With min_pixels=100:
        - Detection 0: 200x150 pixels (both >= 100, should pass)
        - Detection 1: 100x80 pixels (width=100 >=, height=80 <, should fail)
        - Detection 2: 50x40 pixels (both < 100, should fail)
        - Detection 3: 20x20 pixels (both < 100, should fail)
        """
        filter_obj = MinDetectionSizeFilter(min_pixels=100)

        # Only detection 0 should pass
        expected_valid_indices = [0]
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_min_pixels_20_filters_correctly(self, varying_size_detection):
        """Test that min_pixels=20 is very permissive.

        With min_pixels=20:
        - Detection 0: 200x150 pixels (both >= 20, should pass)
        - Detection 1: 100x80 pixels (both >= 20, should pass)
        - Detection 2: 50x40 pixels (both >= 20, should pass)
        - Detection 3: 20x20 pixels (both >= 20, should pass)
        """
        filter_obj = MinDetectionSizeFilter(min_pixels=20)

        # All detections should pass
        assert_all_pass(filter_obj, varying_size_detection)

    def test_min_pixels_10_all_pass(self, varying_size_detection):
        """Test that very low threshold allows all detections."""
        filter_obj = MinDetectionSizeFilter(min_pixels=10)

        # All detections should pass (all have dimensions >= 10)
        assert_all_pass(filter_obj, varying_size_detection)


class TestMinDetectionSizeFilterBothDimensionsRequired:
    """Test that MinDetectionSizeFilter requires BOTH width AND height to meet threshold."""

    def test_only_width_meets_threshold_fails(self, sample_image):
        """Test that detection with only width >= threshold fails.

        Create a detection where width >= min_pixels but height < min_pixels.
        This should fail because BOTH dimensions are required.
        """
        # Create wide but short detection: 150x30
        xyxy = np.array([[100, 200, 250, 230]])  # 150x30 pixels
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.417, 0.391, 0.417, 0.391, 0.479, 0.156, 0.479]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # With min_pixels=100: width=150 >= 100, but height=30 < 100
        filter_obj = MinDetectionSizeFilter(min_pixels=100)
        assert_all_fail(filter_obj, detection)

    def test_only_height_meets_threshold_fails(self, sample_image):
        """Test that detection with only height >= threshold fails.

        Create a detection where height >= min_pixels but width < min_pixels.
        This should fail because BOTH dimensions are required.
        """
        # Create tall but narrow detection: 30x150
        xyxy = np.array([[200, 100, 230, 250]])  # 30x150 pixels
        class_id = np.array([0])
        segmentation_points = [[0.312, 0.208, 0.359, 0.208, 0.359, 0.521, 0.312, 0.521]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # With min_pixels=100: height=150 >= 100, but width=30 < 100
        filter_obj = MinDetectionSizeFilter(min_pixels=100)
        assert_all_fail(filter_obj, detection)

    def test_both_dimensions_meet_threshold_passes(self, sample_image):
        """Test that detection with both dimensions >= threshold passes.

        Create a detection where both width >= min_pixels AND height >= min_pixels.
        This should pass.
        """
        # Create detection with both dimensions >= 100: 150x120
        xyxy = np.array([[100, 100, 250, 220]])  # 150x120 pixels
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.391, 0.208, 0.391, 0.458, 0.156, 0.458]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # With min_pixels=100: width=150 >= 100 AND height=120 >= 100
        filter_obj = MinDetectionSizeFilter(min_pixels=100)
        assert_all_pass(filter_obj, detection)


class TestMinDetectionSizeFilterBoundaryConditions:
    """Test MinDetectionSizeFilter boundary conditions."""

    def test_boundary_both_dimensions_exactly_at_threshold(self, sample_image):
        """Test detection with both dimensions exactly at threshold.

        Create a detection where both width == min_pixels and height == min_pixels.
        This should pass (>= condition).
        """
        # Create 100x100 square detection
        xyxy = np.array([[100, 100, 200, 200]])  # 100x100 pixels
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.312, 0.208, 0.312, 0.417, 0.156, 0.417]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = MinDetectionSizeFilter(min_pixels=100)

        # Both dimensions exactly at threshold should pass
        assert_all_pass(filter_obj, detection)

    def test_boundary_width_just_below_threshold(self, sample_image):
        """Test detection with width just below threshold.

        Width = 99, height = 100, threshold = 100
        Should fail because width < threshold.
        """
        # Create 99x100 detection
        xyxy = np.array([[100, 100, 199, 200]])  # 99x100 pixels
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.311, 0.208, 0.311, 0.417, 0.156, 0.417]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = MinDetectionSizeFilter(min_pixels=100)

        # Width is 99 < 100, should fail
        assert_all_fail(filter_obj, detection)

    def test_boundary_height_just_below_threshold(self, sample_image):
        """Test detection with height just below threshold.

        Width = 100, height = 99, threshold = 100
        Should fail because height < threshold.
        """
        # Create 100x99 detection
        xyxy = np.array([[100, 100, 200, 199]])  # 100x99 pixels
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.312, 0.208, 0.312, 0.415, 0.156, 0.415]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = MinDetectionSizeFilter(min_pixels=100)

        # Height is 99 < 100, should fail
        assert_all_fail(filter_obj, detection)

    def test_boundary_both_just_above_threshold(self, sample_image):
        """Test detection with both dimensions just above threshold.

        Width = 101, height = 101, threshold = 100
        Should pass because both >= threshold.
        """
        # Create 101x101 detection
        xyxy = np.array([[100, 100, 201, 201]])  # 101x101 pixels
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.314, 0.208, 0.314, 0.419, 0.156, 0.419]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = MinDetectionSizeFilter(min_pixels=100)

        # Both dimensions > threshold should pass
        assert_all_pass(filter_obj, detection)


class TestMinDetectionSizeFilterEmptyAndSingleDetection:
    """Test MinDetectionSizeFilter with empty and single detection fixtures."""

    def test_empty_detection(self, empty_detection):
        """Test that empty detection works correctly.

        With no detections, the filter should not raise errors.
        """
        filter_obj = MinDetectionSizeFilter()
        assert len(empty_detection.xyxy) == 0

        # Verify this doesn't raise an error
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []

    def test_single_detection_with_default_threshold(self, single_detection):
        """Test single detection with default 300px threshold.

        single_detection has 128x96 box (both < 300, should fail).
        """
        filter_obj = MinDetectionSizeFilter()  # default min_pixels=300

        # Single detection dimensions are less than 300, should fail
        assert_all_fail(filter_obj, single_detection)

    def test_single_detection_with_lower_threshold(self, single_detection):
        """Test single detection with lower threshold.

        single_detection has 128x96 box.
        With min_pixels=90, should pass (both 128 >= 90 and 96 >= 90).
        """
        filter_obj = MinDetectionSizeFilter(min_pixels=90)

        # Both dimensions >= 90, should pass
        assert_all_pass(filter_obj, single_detection)

    def test_single_detection_boundary(self, single_detection):
        """Test single detection at boundary.

        single_detection has 128x96 box.
        With min_pixels=96, height is exactly at threshold.
        With min_pixels=97, should fail.
        """
        # At threshold
        filter_obj = MinDetectionSizeFilter(min_pixels=96)
        assert_all_pass(filter_obj, single_detection)

        # Just above threshold
        filter_obj = MinDetectionSizeFilter(min_pixels=97)
        assert_all_fail(filter_obj, single_detection)


class TestMinDetectionSizeFilterEdgeCases:
    """Test edge cases for MinDetectionSizeFilter."""

    def test_very_small_threshold(self, varying_size_detection):
        """Test that very small threshold (1 pixel) allows all detections."""
        filter_obj = MinDetectionSizeFilter(min_pixels=1)

        # All detections should pass (all have dimensions >= 1)
        assert_all_pass(filter_obj, varying_size_detection)

    def test_very_large_threshold(self, varying_size_detection):
        """Test that very large threshold (1000 pixels) filters all detections."""
        filter_obj = MinDetectionSizeFilter(min_pixels=1000)

        # All detections should fail (largest is 200x150)
        assert_all_fail(filter_obj, varying_size_detection)

    def test_filter_consistency_across_multiple_calls(self, varying_size_detection):
        """Test that filter returns consistent results across multiple calls.

        Calling is_valid multiple times for the same detection should
        return the same result.
        """
        filter_obj = MinDetectionSizeFilter(min_pixels=50)

        # Call is_valid multiple times for each index
        for index in range(len(varying_size_detection.xyxy)):
            result1 = filter_obj.is_valid(
                varying_size_detection.image, index, varying_size_detection
            )
            result2 = filter_obj.is_valid(
                varying_size_detection.image, index, varying_size_detection
            )
            result3 = filter_obj.is_valid(
                varying_size_detection.image, index, varying_size_detection
            )
            assert result1 == result2 == result3


class TestMinDetectionSizeFilterHelperFunctions:
    """Test MinDetectionSizeFilter using helper functions."""

    def test_count_passing_detections(self, varying_size_detection):
        """Test counting how many detections pass the filter.

        With min_pixels=50, detections 0 and 1 should pass.
        """
        filter_obj = MinDetectionSizeFilter(min_pixels=50)
        count = count_passing_detections(filter_obj, varying_size_detection)
        assert count == 2

    def test_get_passing_indices(self, varying_size_detection):
        """Test getting list of indices that pass the filter.

        With min_pixels=100, only detection 0 should pass.
        """
        filter_obj = MinDetectionSizeFilter(min_pixels=100)
        passing = get_passing_indices(filter_obj, varying_size_detection)
        assert passing == [0]

    def test_get_failing_indices(self, varying_size_detection):
        """Test getting list of indices that fail the filter.

        With min_pixels=100, detections 1, 2, 3 should fail.
        """
        filter_obj = MinDetectionSizeFilter(min_pixels=100)
        failing = get_failing_indices(filter_obj, varying_size_detection)
        assert failing == [1, 2, 3]
