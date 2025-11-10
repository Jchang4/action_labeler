"""Tests for the AspectRatioFilter class.

This module contains comprehensive pytest tests for AspectRatioFilter, which filters
detections based on bounding box aspect ratio (width/height).
"""

import numpy as np
import pytest

from action_labeler.detections.detection import Detection
from action_labeler.filters.aspect_ratio import AspectRatioFilter
from tests.filters.helpers import (
    assert_all_fail,
    assert_all_pass,
    assert_filter_validates_indices,
    count_passing_detections,
    get_failing_indices,
    get_passing_indices,
)


class TestAspectRatioFilterConstructor:
    """Test cases for AspectRatioFilter constructor validation."""

    def test_constructor_with_default_parameters(self):
        """Test that constructor accepts default parameters.

        Default parameters are min_ratio=0.1 and max_ratio=10.0.
        """
        filter_obj = AspectRatioFilter()
        assert filter_obj.min_ratio == 0.1
        assert filter_obj.max_ratio == 10.0

    def test_constructor_with_custom_parameters(self):
        """Test that constructor accepts custom parameters."""
        filter_obj = AspectRatioFilter(min_ratio=0.5, max_ratio=2.0)
        assert filter_obj.min_ratio == 0.5
        assert filter_obj.max_ratio == 2.0

    def test_constructor_rejects_negative_min_ratio(self):
        """Test that constructor rejects negative min_ratio.

        The min_ratio parameter must be non-negative (>= 0).
        """
        with pytest.raises(ValueError, match="min_ratio must be non-negative"):
            AspectRatioFilter(min_ratio=-0.1, max_ratio=10.0)

        with pytest.raises(ValueError, match="min_ratio must be non-negative"):
            AspectRatioFilter(min_ratio=-1.0, max_ratio=10.0)

    def test_constructor_rejects_negative_max_ratio(self):
        """Test that constructor rejects negative max_ratio.

        The max_ratio parameter must be non-negative (>= 0).
        """
        with pytest.raises(ValueError, match="max_ratio must be non-negative"):
            AspectRatioFilter(min_ratio=0.1, max_ratio=-0.1)

        with pytest.raises(ValueError, match="max_ratio must be non-negative"):
            AspectRatioFilter(min_ratio=0.1, max_ratio=-10.0)

    def test_constructor_rejects_min_greater_than_max(self):
        """Test that constructor rejects min_ratio > max_ratio.

        The min_ratio must be less than or equal to max_ratio.
        """
        with pytest.raises(ValueError, match="min_ratio must be <= max_ratio"):
            AspectRatioFilter(min_ratio=5.0, max_ratio=2.0)

        with pytest.raises(ValueError, match="min_ratio must be <= max_ratio"):
            AspectRatioFilter(min_ratio=10.0, max_ratio=0.1)

    def test_constructor_accepts_min_equal_to_max(self):
        """Test that constructor accepts min_ratio equal to max_ratio.

        This creates a filter that only accepts a specific aspect ratio.
        """
        filter_obj = AspectRatioFilter(min_ratio=1.0, max_ratio=1.0)
        assert filter_obj.min_ratio == 1.0
        assert filter_obj.max_ratio == 1.0

    def test_constructor_accepts_zero_min_ratio(self):
        """Test that constructor accepts min_ratio=0."""
        filter_obj = AspectRatioFilter(min_ratio=0.0, max_ratio=10.0)
        assert filter_obj.min_ratio == 0.0
        assert filter_obj.max_ratio == 10.0

    def test_constructor_accepts_zero_max_ratio(self):
        """Test that constructor accepts max_ratio=0 when min_ratio=0."""
        filter_obj = AspectRatioFilter(min_ratio=0.0, max_ratio=0.0)
        assert filter_obj.min_ratio == 0.0
        assert filter_obj.max_ratio == 0.0

    def test_constructor_rejects_both_negative(self):
        """Test that constructor rejects both parameters being negative.

        Should raise ValueError for min_ratio being negative first.
        """
        with pytest.raises(ValueError, match="min_ratio must be non-negative"):
            AspectRatioFilter(min_ratio=-1.0, max_ratio=-0.5)


class TestAspectRatioFilterWithDefaultParameters:
    """Test cases for AspectRatioFilter with default parameters (0.1, 10.0)."""

    def test_default_parameters_with_aspect_ratio_detection(self, aspect_ratio_detection):
        """Test default parameters with the aspect_ratio_detection fixture.

        aspect_ratio_detection has:
        - Detection 0: Square 100x100 (ratio = 1.0) - should pass
        - Detection 1: Wide 200x50 (ratio = 4.0) - should pass
        - Detection 2: Tall 25x100 (ratio = 0.25) - should pass
        - Detection 3: Very wide 320x40 (ratio = 8.0) - should pass

        All detections should pass with default parameters (0.1, 10.0).
        """
        filter_obj = AspectRatioFilter()
        assert_all_pass(filter_obj, aspect_ratio_detection)

    def test_default_parameters_with_single_detection(self, single_detection):
        """Test default parameters with single_detection fixture.

        single_detection has a 128x96 pixel bbox (ratio ≈ 1.33).
        This should pass with default parameters.
        """
        filter_obj = AspectRatioFilter()
        assert_all_pass(filter_obj, single_detection)

    def test_default_parameters_with_empty_detection(self, empty_detection):
        """Test that default parameters work correctly with empty detection.

        With no detections, the filter should not raise errors.
        """
        filter_obj = AspectRatioFilter()
        # Empty detection has no detections to validate
        assert len(empty_detection.xyxy) == 0
        # Verify this doesn't raise an error
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []

    def test_default_parameters_with_varying_size_detection(self, varying_size_detection):
        """Test default parameters with varying_size_detection fixture.

        varying_size_detection has:
        - Detection 0: Large 200x150 (ratio ≈ 1.33) - should pass
        - Detection 1: Medium 100x80 (ratio = 1.25) - should pass
        - Detection 2: Small 50x40 (ratio = 1.25) - should pass
        - Detection 3: Tiny 20x20 (ratio = 1.0) - should pass

        All should pass with default parameters.
        """
        filter_obj = AspectRatioFilter()
        assert_all_pass(filter_obj, varying_size_detection)


class TestAspectRatioFilterSquareObjects:
    """Test cases for filtering square objects (min=0.8, max=1.2)."""

    def test_square_filter_with_aspect_ratio_detection(self, aspect_ratio_detection):
        """Test filtering for roughly square objects.

        aspect_ratio_detection has:
        - Detection 0: Square 100x100 (ratio = 1.0) - should pass
        - Detection 1: Wide 200x50 (ratio = 4.0) - should fail
        - Detection 2: Tall 25x100 (ratio = 0.25) - should fail
        - Detection 3: Very wide 320x40 (ratio = 8.0) - should fail
        """
        filter_obj = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
        expected_valid_indices = [0]  # Only the square detection
        assert_filter_validates_indices(
            filter_obj, aspect_ratio_detection, expected_valid_indices
        )

    def test_square_filter_with_single_detection(self, single_detection):
        """Test square filter with single_detection fixture.

        single_detection has a 128x96 pixel bbox (ratio ≈ 1.33).
        This should fail the square filter (0.8, 1.2).
        """
        filter_obj = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
        assert_all_fail(filter_obj, single_detection)

    def test_square_filter_with_varying_size_detection(self, varying_size_detection):
        """Test square filter with varying_size_detection fixture.

        varying_size_detection has:
        - Detection 0: Large 200x150 (ratio ≈ 1.33) - should fail
        - Detection 1: Medium 100x80 (ratio = 1.25) - should fail
        - Detection 2: Small 50x40 (ratio = 1.25) - should fail
        - Detection 3: Tiny 20x20 (ratio = 1.0) - should pass

        Only the tiny square detection should pass.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
        expected_valid_indices = [3]  # Only the tiny 20x20 square
        assert_filter_validates_indices(
            filter_obj, varying_size_detection, expected_valid_indices
        )

    def test_exact_square_filter(self, aspect_ratio_detection):
        """Test filtering for exactly square objects (min=1.0, max=1.0).

        Only detections with aspect ratio exactly 1.0 should pass.
        """
        filter_obj = AspectRatioFilter(min_ratio=1.0, max_ratio=1.0)
        expected_valid_indices = [0]  # Only the 100x100 square
        assert_filter_validates_indices(
            filter_obj, aspect_ratio_detection, expected_valid_indices
        )


class TestAspectRatioFilterWideObjects:
    """Test cases for filtering wide objects (min=3.0, max=10.0)."""

    def test_wide_filter_with_aspect_ratio_detection(self, aspect_ratio_detection):
        """Test filtering for wide objects.

        aspect_ratio_detection has:
        - Detection 0: Square 100x100 (ratio = 1.0) - should fail
        - Detection 1: Wide 200x50 (ratio = 4.0) - should pass
        - Detection 2: Tall 25x100 (ratio = 0.25) - should fail
        - Detection 3: Very wide 320x40 (ratio = 8.0) - should pass
        """
        filter_obj = AspectRatioFilter(min_ratio=3.0, max_ratio=10.0)
        expected_valid_indices = [1, 3]  # Wide and very wide detections
        assert_filter_validates_indices(
            filter_obj, aspect_ratio_detection, expected_valid_indices
        )

    def test_wide_filter_excludes_square_and_tall(self, aspect_ratio_detection):
        """Test that wide filter excludes square and tall objects.

        Only detections with ratio >= 3.0 should pass.
        """
        filter_obj = AspectRatioFilter(min_ratio=3.0, max_ratio=10.0)
        failing = get_failing_indices(filter_obj, aspect_ratio_detection)
        assert failing == [0, 2]  # Square and tall detections

    def test_very_wide_filter(self, aspect_ratio_detection):
        """Test filtering for very wide objects (min=5.0, max=10.0).

        aspect_ratio_detection has:
        - Detection 1: Wide 200x50 (ratio = 4.0) - should fail
        - Detection 3: Very wide 320x40 (ratio = 8.0) - should pass
        """
        filter_obj = AspectRatioFilter(min_ratio=5.0, max_ratio=10.0)
        expected_valid_indices = [3]  # Only very wide detection
        assert_filter_validates_indices(
            filter_obj, aspect_ratio_detection, expected_valid_indices
        )

    def test_wide_filter_with_single_detection(self, single_detection):
        """Test wide filter with single_detection fixture.

        single_detection has ratio ≈ 1.33, which should fail wide filter.
        """
        filter_obj = AspectRatioFilter(min_ratio=3.0, max_ratio=10.0)
        assert_all_fail(filter_obj, single_detection)


class TestAspectRatioFilterTallObjects:
    """Test cases for filtering tall objects (min=0.1, max=0.5)."""

    def test_tall_filter_with_aspect_ratio_detection(self, aspect_ratio_detection):
        """Test filtering for tall objects.

        aspect_ratio_detection has:
        - Detection 0: Square 100x100 (ratio = 1.0) - should fail
        - Detection 1: Wide 200x50 (ratio = 4.0) - should fail
        - Detection 2: Tall 25x100 (ratio = 0.25) - should pass
        - Detection 3: Very wide 320x40 (ratio = 8.0) - should fail
        """
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=0.5)
        expected_valid_indices = [2]  # Only the tall detection
        assert_filter_validates_indices(
            filter_obj, aspect_ratio_detection, expected_valid_indices
        )

    def test_tall_filter_excludes_square_and_wide(self, aspect_ratio_detection):
        """Test that tall filter excludes square and wide objects.

        Only detections with ratio <= 0.5 should pass.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=0.5)
        failing = get_failing_indices(filter_obj, aspect_ratio_detection)
        assert failing == [0, 1, 3]  # Square and wide detections

    def test_very_tall_filter(self, aspect_ratio_detection):
        """Test filtering for very tall objects (min=0.1, max=0.3).

        aspect_ratio_detection has:
        - Detection 2: Tall 25x100 (ratio = 0.25) - should pass
        """
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=0.3)
        expected_valid_indices = [2]  # Only the tall detection
        assert_filter_validates_indices(
            filter_obj, aspect_ratio_detection, expected_valid_indices
        )

    def test_tall_filter_with_single_detection(self, single_detection):
        """Test tall filter with single_detection fixture.

        single_detection has ratio ≈ 1.33, which should fail tall filter.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=0.5)
        assert_all_fail(filter_obj, single_detection)


class TestAspectRatioFilterBoundaryConditions:
    """Test boundary conditions for AspectRatioFilter."""

    def test_boundary_exact_min_ratio(self, aspect_ratio_detection):
        """Test that detections with aspect ratio exactly equal to min_ratio pass.

        Detection 2 has ratio 0.25, so min_ratio=0.25 should include it.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.25, max_ratio=10.0)
        # Detection 2 (ratio=0.25) should pass at the boundary
        result = filter_obj.is_valid(
            aspect_ratio_detection.image, 2, aspect_ratio_detection
        )
        assert result is True

    def test_boundary_exact_max_ratio(self, aspect_ratio_detection):
        """Test that detections with aspect ratio exactly equal to max_ratio pass.

        Detection 1 has ratio 4.0, so max_ratio=4.0 should include it.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=4.0)
        # Detection 1 (ratio=4.0) should pass at the boundary
        result = filter_obj.is_valid(
            aspect_ratio_detection.image, 1, aspect_ratio_detection
        )
        assert result is True

    def test_boundary_just_below_min_ratio(self, aspect_ratio_detection):
        """Test that detections just below min_ratio fail.

        Detection 2 has ratio 0.25, so min_ratio=0.26 should exclude it.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.26, max_ratio=10.0)
        # Detection 2 (ratio=0.25) should fail just below the boundary
        result = filter_obj.is_valid(
            aspect_ratio_detection.image, 2, aspect_ratio_detection
        )
        assert result is False

    def test_boundary_just_above_max_ratio(self, aspect_ratio_detection):
        """Test that detections just above max_ratio fail.

        Detection 1 has ratio 4.0, so max_ratio=3.9 should exclude it.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=3.9)
        # Detection 1 (ratio=4.0) should fail just above the boundary
        result = filter_obj.is_valid(
            aspect_ratio_detection.image, 1, aspect_ratio_detection
        )
        assert result is False

    def test_boundary_min_ratio_zero(self, aspect_ratio_detection):
        """Test that min_ratio=0 allows all aspect ratios (if max is high enough).

        With min_ratio=0 and max_ratio=10.0, all normal detections should pass.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.0, max_ratio=10.0)
        assert_all_pass(filter_obj, aspect_ratio_detection)

    def test_boundary_max_ratio_zero_rejects_all(self, aspect_ratio_detection):
        """Test that max_ratio=0 rejects all detections with width > 0.

        With min_ratio=0 and max_ratio=0, only zero-width detections would pass.
        All normal detections should fail.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.0, max_ratio=0.0)
        assert_all_fail(filter_obj, aspect_ratio_detection)

    def test_boundary_both_equal_to_aspect_ratio(self, aspect_ratio_detection):
        """Test that min=max=ratio allows only that specific ratio.

        Detection 1 has ratio 4.0, so min=max=4.0 should pass only detection 1.
        """
        filter_obj = AspectRatioFilter(min_ratio=4.0, max_ratio=4.0)
        expected_valid_indices = [1]
        assert_filter_validates_indices(
            filter_obj, aspect_ratio_detection, expected_valid_indices
        )


class TestAspectRatioFilterEdgeCases:
    """Test edge cases for AspectRatioFilter."""

    def test_zero_height_detection_rejected(self, sample_image):
        """Test that detections with height=0 are rejected to avoid division by zero.

        Create a custom detection with height=0 and verify it fails.
        """
        # Create a detection with zero height: y1 == y2
        xyxy = np.array([[100, 200, 300, 200]])  # Width=200, height=0
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.417, 0.469, 0.417, 0.469, 0.417, 0.156, 0.417]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = AspectRatioFilter()
        # Detection with height=0 should always fail
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is False

    def test_zero_width_detection(self, sample_image):
        """Test detection with width=0 (aspect ratio = 0).

        This should pass filters with min_ratio=0 but fail most others.
        """
        # Create a detection with zero width: x1 == x2
        xyxy = np.array([[200, 100, 200, 300]])  # Width=0, height=200
        class_id = np.array([0])
        segmentation_points = [[0.312, 0.208, 0.312, 0.208, 0.312, 0.625, 0.312, 0.625]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # With default parameters (min=0.1), should fail (ratio=0 < 0.1)
        filter_obj = AspectRatioFilter()
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is False

        # With min_ratio=0, should pass
        filter_obj = AspectRatioFilter(min_ratio=0.0, max_ratio=10.0)
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is True

    def test_very_large_aspect_ratio(self, sample_image):
        """Test detection with very large aspect ratio.

        Create a detection with width=1000, height=1 (ratio=1000.0).
        """
        xyxy = np.array([[10, 100, 1010, 101]])  # Width=1000, height=1
        class_id = np.array([0])
        segmentation_points = [[0.016, 0.208, 0.578, 0.208, 0.578, 0.210, 0.016, 0.210]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Default parameters (max=10.0) should fail
        filter_obj = AspectRatioFilter()
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is False

        # With max_ratio=1000.0, should pass
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=1000.0)
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is True

    def test_very_small_aspect_ratio(self, sample_image):
        """Test detection with very small aspect ratio.

        Create a detection with width=1, height=1000 (ratio=0.001).
        """
        xyxy = np.array([[100, 10, 101, 1010]])  # Width=1, height=1000
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.021, 0.158, 0.021, 0.158, 0.521, 0.156, 0.521]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Default parameters (min=0.1) should fail
        filter_obj = AspectRatioFilter()
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is False

        # With min_ratio=0.001, should pass
        filter_obj = AspectRatioFilter(min_ratio=0.001, max_ratio=10.0)
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is True

    def test_single_pixel_detection(self, sample_image):
        """Test detection with 1x1 pixel (aspect ratio = 1.0).

        This is a valid edge case that should pass square filters.
        """
        xyxy = np.array([[100, 100, 101, 101]])  # Width=1, height=1
        class_id = np.array([0])
        segmentation_points = [[0.156, 0.208, 0.158, 0.208, 0.158, 0.210, 0.156, 0.210]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Should pass square filter
        filter_obj = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is True

        # Should pass default filter
        filter_obj = AspectRatioFilter()
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is True


class TestAspectRatioFilterHelperFunctions:
    """Test AspectRatioFilter using helper functions to verify integration."""

    def test_count_passing_detections(self, aspect_ratio_detection):
        """Test counting how many detections pass the filter.

        aspect_ratio_detection with square filter (0.8, 1.2) should have 1 passing.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
        count = count_passing_detections(filter_obj, aspect_ratio_detection)
        assert count == 1

    def test_get_passing_indices(self, aspect_ratio_detection):
        """Test getting list of indices that pass the filter.

        Wide filter (3.0, 10.0) should return indices [1, 3].
        """
        filter_obj = AspectRatioFilter(min_ratio=3.0, max_ratio=10.0)
        passing = get_passing_indices(filter_obj, aspect_ratio_detection)
        assert passing == [1, 3]

    def test_get_failing_indices(self, aspect_ratio_detection):
        """Test getting list of indices that fail the filter.

        Tall filter (0.1, 0.5) should have failing indices [0, 1, 3].
        """
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=0.5)
        failing = get_failing_indices(filter_obj, aspect_ratio_detection)
        assert failing == [0, 1, 3]

    def test_assert_filter_validates_indices_helper(self, aspect_ratio_detection):
        """Test that assert_filter_validates_indices works correctly.

        This verifies the integration between AspectRatioFilter and test helpers.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
        expected_valid_indices = [0]  # Only square detection

        # This should not raise an assertion error
        assert_filter_validates_indices(
            filter_obj, aspect_ratio_detection, expected_valid_indices
        )

        # This should raise an assertion error (incorrect expectation)
        with pytest.raises(AssertionError):
            assert_filter_validates_indices(
                filter_obj, aspect_ratio_detection, [1, 2, 3]
            )


class TestAspectRatioFilterConsistency:
    """Test consistency and reliability of AspectRatioFilter."""

    def test_filter_consistency_across_multiple_calls(self, aspect_ratio_detection):
        """Test that filter returns consistent results across multiple calls.

        Calling is_valid multiple times for the same detection should
        return the same result.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)

        # Call is_valid multiple times for each index
        for index in range(len(aspect_ratio_detection.xyxy)):
            result1 = filter_obj.is_valid(
                aspect_ratio_detection.image, index, aspect_ratio_detection
            )
            result2 = filter_obj.is_valid(
                aspect_ratio_detection.image, index, aspect_ratio_detection
            )
            result3 = filter_obj.is_valid(
                aspect_ratio_detection.image, index, aspect_ratio_detection
            )
            assert result1 == result2 == result3

    def test_filter_with_different_images_same_detections(
        self, aspect_ratio_detection, sample_image, small_image
    ):
        """Test that filter works consistently regardless of image.

        The filter should only care about bbox dimensions, not the image itself.
        """
        filter_obj = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)

        # Test with original image
        result1 = filter_obj.is_valid(
            aspect_ratio_detection.image, 0, aspect_ratio_detection
        )

        # Test with different images (should give same result)
        result2 = filter_obj.is_valid(sample_image, 0, aspect_ratio_detection)
        result3 = filter_obj.is_valid(small_image, 0, aspect_ratio_detection)

        assert result1 == result2 == result3 == True

    def test_multiple_filters_independent(self, aspect_ratio_detection):
        """Test that multiple filter instances are independent.

        Creating multiple filters should not affect each other.
        """
        filter1 = AspectRatioFilter(min_ratio=0.8, max_ratio=1.2)
        filter2 = AspectRatioFilter(min_ratio=3.0, max_ratio=10.0)
        filter3 = AspectRatioFilter(min_ratio=0.1, max_ratio=0.5)

        # Each filter should have different results
        result1 = filter1.is_valid(
            aspect_ratio_detection.image, 0, aspect_ratio_detection
        )
        result2 = filter2.is_valid(
            aspect_ratio_detection.image, 0, aspect_ratio_detection
        )
        result3 = filter3.is_valid(
            aspect_ratio_detection.image, 0, aspect_ratio_detection
        )

        # Detection 0 is square (ratio=1.0)
        assert result1 is True  # Passes square filter
        assert result2 is False  # Fails wide filter
        assert result3 is False  # Fails tall filter

    def test_aspect_ratio_calculation_precision(self, sample_image):
        """Test that aspect ratio calculation maintains precision.

        Verify that float division is used correctly.
        """
        # Create detection with specific dimensions
        # Width=100, Height=3 -> ratio = 33.333...
        xyxy = np.array([[10, 10, 110, 13]])
        class_id = np.array([0])
        segmentation_points = [[0.016, 0.021, 0.172, 0.021, 0.172, 0.027, 0.016, 0.027]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Should pass with max=33.4
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=33.4)
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is True

        # Should fail with max=33.3
        filter_obj = AspectRatioFilter(min_ratio=0.1, max_ratio=33.3)
        result = filter_obj.is_valid(sample_image, 0, detection)
        assert result is False
