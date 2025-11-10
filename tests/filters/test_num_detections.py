"""Tests for the num_detections filter classes.

This module tests three filter classes that validate images based on the
number of detections present:
- SingleDetectionFilter: Validates images with exactly 1 detection
- MaxDetectionsFilter: Validates images with at most N detections
- MinDetectionsFilter: Validates images with at least N detections

Note: These filters operate at the IMAGE level, not the DETECTION level.
This means the filter returns the same validation result for ALL detection
indices in a given image.
"""

import pytest

from action_labeler.filters.num_detections import (
    MaxDetectionsFilter,
    MinDetectionsFilter,
    SingleDetectionFilter,
)
from tests.filters.helpers import assert_all_fail, assert_all_pass


class TestSingleDetectionFilter:
    """Test cases for SingleDetectionFilter.

    SingleDetectionFilter validates images that contain exactly one detection.
    Images with 0, 2, or more detections will fail validation.
    """

    def test_empty_detection_fails(self, empty_detection):
        """Test that images with 0 detections fail validation.

        An empty detection (no detections) should fail because the filter
        requires exactly 1 detection.
        """
        filter_obj = SingleDetectionFilter()

        # Empty detection has no indices to check, but we verify the behavior
        # by calling is_valid with index 0 (which won't exist)
        result = filter_obj.is_valid(
            empty_detection.image, 0, empty_detection
        )
        assert result is False

    def test_single_detection_passes(self, single_detection):
        """Test that images with exactly 1 detection pass validation.

        A detection with exactly one item should pass. Since this is an
        image-level filter, the result is the same regardless of which
        index we check.
        """
        filter_obj = SingleDetectionFilter()

        # For single detection, all indices (just 0) should pass
        assert_all_pass(filter_obj, single_detection)

    def test_multiple_detections_fail(self, multi_class_detection):
        """Test that images with multiple detections fail validation.

        A detection with 5 items should fail because the filter requires
        exactly 1 detection. Since this is an image-level filter, ALL
        indices (0-4) should fail.
        """
        filter_obj = SingleDetectionFilter()

        # All 5 detections should fail because the image has more than 1 detection
        assert_all_fail(filter_obj, multi_class_detection)

    def test_four_detections_fail(self, varying_size_detection):
        """Test that images with 4 detections fail validation.

        Testing with a different fixture to ensure the filter consistently
        rejects any image with more than 1 detection.
        """
        filter_obj = SingleDetectionFilter()

        # All 4 detections should fail
        assert_all_fail(filter_obj, varying_size_detection)

    def test_filter_is_image_level(self, single_detection):
        """Test that the filter validation is consistent across all indices.

        Since this is an image-level filter, all indices in the same image
        should return the same validation result.
        """
        filter_obj = SingleDetectionFilter()

        # For single detection, index 0 should pass
        result = filter_obj.is_valid(
            single_detection.image, 0, single_detection
        )
        assert result is True

    def test_filter_is_image_level_multi(self, multi_class_detection):
        """Test that all indices fail for multi-detection images.

        All 5 indices should return False since the image has 5 detections.
        """
        filter_obj = SingleDetectionFilter()

        # Check that all indices return the same result (False)
        for i in range(len(multi_class_detection.xyxy)):
            result = filter_obj.is_valid(
                multi_class_detection.image, i, multi_class_detection
            )
            assert result is False


class TestMaxDetectionsFilter:
    """Test cases for MaxDetectionsFilter.

    MaxDetectionsFilter validates images that contain at most N detections.
    Images with more than N detections will fail validation.
    """

    def test_constructor_validates_positive_max(self):
        """Test that constructor rejects max_detections <= 0.

        The max_detections parameter must be greater than 0. Values of 0
        or negative numbers should raise a ValueError.
        """
        with pytest.raises(ValueError, match="max_detections must be greater than 0"):
            MaxDetectionsFilter(max_detections=0)

        with pytest.raises(ValueError, match="max_detections must be greater than 0"):
            MaxDetectionsFilter(max_detections=-1)

        with pytest.raises(ValueError, match="max_detections must be greater than 0"):
            MaxDetectionsFilter(max_detections=-10)

    def test_max_1_single_detection_passes(self, single_detection):
        """Test that max_detections=1 passes for images with 1 detection.

        When max_detections is 1, an image with exactly 1 detection should
        pass validation.
        """
        filter_obj = MaxDetectionsFilter(max_detections=1)

        # Single detection (1 detection) should pass
        assert_all_pass(filter_obj, single_detection)

    def test_max_1_multiple_detections_fail(self, multi_class_detection):
        """Test that max_detections=1 fails for images with multiple detections.

        When max_detections is 1, an image with 5 detections should fail
        validation. All 5 indices should fail since this is an image-level filter.
        """
        filter_obj = MaxDetectionsFilter(max_detections=1)

        # Multi-class detection (5 detections) should fail when max is 1
        assert_all_fail(filter_obj, multi_class_detection)

    def test_max_5_exactly_5_detections_passes(self, multi_class_detection):
        """Test boundary condition: max_detections=5 with exactly 5 detections.

        When max_detections is 5, an image with exactly 5 detections should
        pass validation (boundary case).
        """
        filter_obj = MaxDetectionsFilter(max_detections=5)

        # Multi-class detection has exactly 5 detections, should pass
        assert_all_pass(filter_obj, multi_class_detection)

    def test_max_10_with_5_detections_passes(self, multi_class_detection):
        """Test that max_detections=10 passes for images with 5 detections.

        When max_detections is 10, an image with 5 detections should pass
        since 5 <= 10.
        """
        filter_obj = MaxDetectionsFilter(max_detections=10)

        # Multi-class detection (5 detections) should pass when max is 10
        assert_all_pass(filter_obj, multi_class_detection)

    def test_max_4_with_5_detections_fails(self, multi_class_detection):
        """Test that max_detections=4 fails for images with 5 detections.

        When max_detections is 4, an image with 5 detections should fail
        since 5 > 4. All indices should fail.
        """
        filter_obj = MaxDetectionsFilter(max_detections=4)

        # Multi-class detection (5 detections) should fail when max is 4
        assert_all_fail(filter_obj, multi_class_detection)

    def test_empty_detection_passes(self, empty_detection):
        """Test that empty detections pass max filter.

        An image with 0 detections should pass any max_detections filter
        since 0 is less than or equal to any positive max value.
        """
        filter_obj = MaxDetectionsFilter(max_detections=1)

        # Empty detection (0 detections) should pass (0 <= 1)
        result = filter_obj.is_valid(
            empty_detection.image, 0, empty_detection
        )
        assert result is True

    def test_max_4_with_4_detections_passes(self, varying_size_detection):
        """Test boundary condition: max_detections=4 with exactly 4 detections.

        When max_detections is 4, an image with exactly 4 detections should
        pass validation (boundary case).
        """
        filter_obj = MaxDetectionsFilter(max_detections=4)

        # Varying size detection has exactly 4 detections, should pass
        assert_all_pass(filter_obj, varying_size_detection)

    def test_max_3_with_4_detections_fails(self, varying_size_detection):
        """Test that max_detections=3 fails for images with 4 detections.

        When max_detections is 3, an image with 4 detections should fail
        since 4 > 3.
        """
        filter_obj = MaxDetectionsFilter(max_detections=3)

        # Varying size detection (4 detections) should fail when max is 3
        assert_all_fail(filter_obj, varying_size_detection)

    def test_filter_is_image_level(self, multi_class_detection):
        """Test that all indices return the same validation result.

        Since this is an image-level filter, all 5 indices should return
        the same result for the same image.
        """
        filter_obj = MaxDetectionsFilter(max_detections=10)

        # All indices should pass
        for i in range(len(multi_class_detection.xyxy)):
            result = filter_obj.is_valid(
                multi_class_detection.image, i, multi_class_detection
            )
            assert result is True

        # Now test with max=2, all indices should fail
        filter_obj = MaxDetectionsFilter(max_detections=2)
        for i in range(len(multi_class_detection.xyxy)):
            result = filter_obj.is_valid(
                multi_class_detection.image, i, multi_class_detection
            )
            assert result is False


class TestMinDetectionsFilter:
    """Test cases for MinDetectionsFilter.

    MinDetectionsFilter validates images that contain at least N detections.
    Images with fewer than N detections will fail validation.
    """

    def test_constructor_validates_positive_min(self):
        """Test that constructor rejects min_detections <= 0.

        The min_detections parameter must be greater than 0. Values of 0
        or negative numbers should raise a ValueError.
        """
        with pytest.raises(ValueError, match="min_detections must be greater than 0"):
            MinDetectionsFilter(min_detections=0)

        with pytest.raises(ValueError, match="min_detections must be greater than 0"):
            MinDetectionsFilter(min_detections=-1)

        with pytest.raises(ValueError, match="min_detections must be greater than 0"):
            MinDetectionsFilter(min_detections=-10)

    def test_min_1_empty_detection_fails(self, empty_detection):
        """Test that min_detections=1 fails for images with 0 detections.

        When min_detections is 1, an image with 0 detections should fail
        since 0 < 1.
        """
        filter_obj = MinDetectionsFilter(min_detections=1)

        # Empty detection (0 detections) should fail when min is 1
        result = filter_obj.is_valid(
            empty_detection.image, 0, empty_detection
        )
        assert result is False

    def test_min_1_single_detection_passes(self, single_detection):
        """Test that min_detections=1 passes for images with 1 detection.

        When min_detections is 1, an image with exactly 1 detection should
        pass validation (boundary case).
        """
        filter_obj = MinDetectionsFilter(min_detections=1)

        # Single detection (1 detection) should pass when min is 1
        assert_all_pass(filter_obj, single_detection)

    def test_min_5_exactly_5_detections_passes(self, multi_class_detection):
        """Test boundary condition: min_detections=5 with exactly 5 detections.

        When min_detections is 5, an image with exactly 5 detections should
        pass validation (boundary case).
        """
        filter_obj = MinDetectionsFilter(min_detections=5)

        # Multi-class detection has exactly 5 detections, should pass
        assert_all_pass(filter_obj, multi_class_detection)

    def test_min_10_with_5_detections_fails(self, multi_class_detection):
        """Test that min_detections=10 fails for images with 5 detections.

        When min_detections is 10, an image with 5 detections should fail
        since 5 < 10. All 5 indices should fail.
        """
        filter_obj = MinDetectionsFilter(min_detections=10)

        # Multi-class detection (5 detections) should fail when min is 10
        assert_all_fail(filter_obj, multi_class_detection)

    def test_min_3_with_5_detections_passes(self, multi_class_detection):
        """Test that min_detections=3 passes for images with 5 detections.

        When min_detections is 3, an image with 5 detections should pass
        since 5 >= 3.
        """
        filter_obj = MinDetectionsFilter(min_detections=3)

        # Multi-class detection (5 detections) should pass when min is 3
        assert_all_pass(filter_obj, multi_class_detection)

    def test_min_6_with_5_detections_fails(self, multi_class_detection):
        """Test that min_detections=6 fails for images with 5 detections.

        When min_detections is 6, an image with 5 detections should fail
        since 5 < 6.
        """
        filter_obj = MinDetectionsFilter(min_detections=6)

        # Multi-class detection (5 detections) should fail when min is 6
        assert_all_fail(filter_obj, multi_class_detection)

    def test_min_4_with_4_detections_passes(self, varying_size_detection):
        """Test boundary condition: min_detections=4 with exactly 4 detections.

        When min_detections is 4, an image with exactly 4 detections should
        pass validation (boundary case).
        """
        filter_obj = MinDetectionsFilter(min_detections=4)

        # Varying size detection has exactly 4 detections, should pass
        assert_all_pass(filter_obj, varying_size_detection)

    def test_min_5_with_4_detections_fails(self, varying_size_detection):
        """Test that min_detections=5 fails for images with 4 detections.

        When min_detections is 5, an image with 4 detections should fail
        since 4 < 5.
        """
        filter_obj = MinDetectionsFilter(min_detections=5)

        # Varying size detection (4 detections) should fail when min is 5
        assert_all_fail(filter_obj, varying_size_detection)

    def test_min_2_with_single_detection_fails(self, single_detection):
        """Test that min_detections=2 fails for images with 1 detection.

        When min_detections is 2, an image with 1 detection should fail
        since 1 < 2.
        """
        filter_obj = MinDetectionsFilter(min_detections=2)

        # Single detection (1 detection) should fail when min is 2
        assert_all_fail(filter_obj, single_detection)

    def test_filter_is_image_level(self, multi_class_detection):
        """Test that all indices return the same validation result.

        Since this is an image-level filter, all 5 indices should return
        the same result for the same image.
        """
        filter_obj = MinDetectionsFilter(min_detections=3)

        # All indices should pass (5 >= 3)
        for i in range(len(multi_class_detection.xyxy)):
            result = filter_obj.is_valid(
                multi_class_detection.image, i, multi_class_detection
            )
            assert result is True

        # Now test with min=10, all indices should fail (5 < 10)
        filter_obj = MinDetectionsFilter(min_detections=10)
        for i in range(len(multi_class_detection.xyxy)):
            result = filter_obj.is_valid(
                multi_class_detection.image, i, multi_class_detection
            )
            assert result is False
