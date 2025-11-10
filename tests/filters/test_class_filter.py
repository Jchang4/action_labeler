"""Tests for the ClassFilter class.

This module contains comprehensive pytest tests for ClassFilter, which filters
detections based on class IDs using allowed or excluded class lists.
"""

import pytest

from action_labeler.filters.class_filter import ClassFilter
from tests.filters.helpers import (
    assert_all_fail,
    assert_all_pass,
    assert_filter_validates_indices,
    count_passing_detections,
    get_failing_indices,
    get_passing_indices,
)


class TestClassFilterConstructor:
    """Test cases for ClassFilter constructor validation."""

    def test_constructor_with_both_allowed_and_excluded_raises_error(self):
        """Test that providing both allowed_classes and excluded_classes raises ValueError.

        The ClassFilter should not allow both parameters to be set simultaneously
        as this would create ambiguous filtering logic.
        """
        with pytest.raises(ValueError) as excinfo:
            ClassFilter(allowed_classes=[0, 1], excluded_classes=[2, 3])

        assert "Cannot specify both allowed_classes and excluded_classes" in str(
            excinfo.value
        )

    def test_constructor_with_only_allowed_classes(self):
        """Test that constructor accepts only allowed_classes parameter."""
        filter_obj = ClassFilter(allowed_classes=[0, 1, 2])
        assert filter_obj.allowed_classes == {0, 1, 2}
        assert filter_obj.excluded_classes is None

    def test_constructor_with_only_excluded_classes(self):
        """Test that constructor accepts only excluded_classes parameter."""
        filter_obj = ClassFilter(excluded_classes=[3, 4, 5])
        assert filter_obj.allowed_classes is None
        assert filter_obj.excluded_classes == {3, 4, 5}

    def test_constructor_with_neither_parameter(self):
        """Test that constructor accepts neither parameter (both None)."""
        filter_obj = ClassFilter()
        assert filter_obj.allowed_classes is None
        assert filter_obj.excluded_classes is None

    def test_constructor_converts_lists_to_sets(self):
        """Test that constructor converts list inputs to sets.

        This ensures efficient membership testing and removes duplicates.
        """
        filter_obj = ClassFilter(allowed_classes=[0, 1, 2, 2, 1])
        assert filter_obj.allowed_classes == {0, 1, 2}
        assert len(filter_obj.allowed_classes) == 3


class TestClassFilterWithAllowedClasses:
    """Test cases for ClassFilter with allowed_classes parameter."""

    def test_allowed_classes_filters_correctly(self, multi_class_detection):
        """Test that only specified allowed classes pass the filter.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With allowed_classes=[0, 1], only indices with class 0 or 1 should pass.
        """
        filter_obj = ClassFilter(allowed_classes=[0, 1])
        # Indices 0, 1, 3, 4 have classes 0, 1, 0, 1 respectively (should pass)
        # Index 2 has class 2 (should fail)
        expected_valid_indices = [0, 1, 3, 4]
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_allowed_classes_single_class(self, multi_class_detection):
        """Test filtering to allow only a single class.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With allowed_classes=[0], only indices 0 and 3 should pass.
        """
        filter_obj = ClassFilter(allowed_classes=[0])
        expected_valid_indices = [0, 3]  # Only class 0
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_allowed_classes_all_classes(self, multi_class_detection):
        """Test that allowing all existing classes lets all detections pass.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With allowed_classes=[0, 1, 2], all detections should pass.
        """
        filter_obj = ClassFilter(allowed_classes=[0, 1, 2])
        assert_all_pass(filter_obj, multi_class_detection)

    def test_allowed_classes_none_present(self, multi_class_detection):
        """Test that specifying classes not in the detection filters all out.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With allowed_classes=[5, 6], no detections should pass.
        """
        filter_obj = ClassFilter(allowed_classes=[5, 6])
        assert_all_fail(filter_obj, multi_class_detection)

    def test_allowed_classes_with_single_detection(self, single_detection):
        """Test allowed_classes with a detection containing only one class.

        single_detection has class: [0]
        """
        # Should pass when class is allowed
        filter_obj = ClassFilter(allowed_classes=[0])
        assert_all_pass(filter_obj, single_detection)

        # Should fail when class is not allowed
        filter_obj = ClassFilter(allowed_classes=[1, 2])
        assert_all_fail(filter_obj, single_detection)

    def test_allowed_classes_empty_list(self, multi_class_detection):
        """Test that an empty allowed_classes list is treated as None.

        Due to the implementation, an empty list is falsy and converted to None,
        which means all classes are allowed. This is a quirk of the current
        implementation.
        """
        filter_obj = ClassFilter(allowed_classes=[])
        # Empty list is converted to None, so all classes pass
        assert filter_obj.allowed_classes is None
        assert_all_pass(filter_obj, multi_class_detection)

    def test_allowed_classes_with_empty_detection(self, empty_detection):
        """Test that allowed_classes works correctly with empty detection.

        With no detections, the filter should not raise errors.
        """
        filter_obj = ClassFilter(allowed_classes=[0, 1])
        # Empty detection has no detections to validate
        assert len(empty_detection.xyxy) == 0
        # Verify this doesn't raise an error
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestClassFilterWithExcludedClasses:
    """Test cases for ClassFilter with excluded_classes parameter."""

    def test_excluded_classes_filters_correctly(self, multi_class_detection):
        """Test that specified excluded classes fail the filter.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With excluded_classes=[2], only index 2 should fail.
        """
        filter_obj = ClassFilter(excluded_classes=[2])
        # All indices pass except index 2 (class 2)
        expected_valid_indices = [0, 1, 3, 4]
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_excluded_classes_multiple_classes(self, multi_class_detection):
        """Test excluding multiple classes.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With excluded_classes=[0, 2], only indices 1 and 4 (class 1) should pass.
        """
        filter_obj = ClassFilter(excluded_classes=[0, 2])
        expected_valid_indices = [1, 4]  # Only class 1
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_excluded_classes_all_classes(self, multi_class_detection):
        """Test that excluding all existing classes filters all detections.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With excluded_classes=[0, 1, 2], no detections should pass.
        """
        filter_obj = ClassFilter(excluded_classes=[0, 1, 2])
        assert_all_fail(filter_obj, multi_class_detection)

    def test_excluded_classes_none_present(self, multi_class_detection):
        """Test that excluding classes not in the detection allows all to pass.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With excluded_classes=[5, 6], all detections should pass.
        """
        filter_obj = ClassFilter(excluded_classes=[5, 6])
        assert_all_pass(filter_obj, multi_class_detection)

    def test_excluded_classes_with_single_detection(self, single_detection):
        """Test excluded_classes with a detection containing only one class.

        single_detection has class: [0]
        """
        # Should fail when class is excluded
        filter_obj = ClassFilter(excluded_classes=[0])
        assert_all_fail(filter_obj, single_detection)

        # Should pass when class is not excluded
        filter_obj = ClassFilter(excluded_classes=[1, 2])
        assert_all_pass(filter_obj, single_detection)

    def test_excluded_classes_empty_list(self, multi_class_detection):
        """Test that an empty excluded_classes list is treated as None.

        Due to the implementation, an empty list is falsy and converted to None,
        which means all classes are allowed (none are excluded).
        """
        filter_obj = ClassFilter(excluded_classes=[])
        # Empty list is converted to None, so all classes pass
        assert filter_obj.excluded_classes is None
        assert_all_pass(filter_obj, multi_class_detection)

    def test_excluded_classes_with_empty_detection(self, empty_detection):
        """Test that excluded_classes works correctly with empty detection.

        With no detections, the filter should not raise errors.
        """
        filter_obj = ClassFilter(excluded_classes=[0, 1])
        # Empty detection has no detections to validate
        assert len(empty_detection.xyxy) == 0
        # Verify this doesn't raise an error
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestClassFilterWithNeither:
    """Test cases for ClassFilter when neither allowed nor excluded is specified."""

    def test_neither_parameter_allows_all(self, multi_class_detection):
        """Test that when neither parameter is set, all detections pass.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With no restrictions, all should pass.
        """
        filter_obj = ClassFilter()
        assert_all_pass(filter_obj, multi_class_detection)

    def test_neither_parameter_with_single_detection(self, single_detection):
        """Test that single detection passes with no restrictions."""
        filter_obj = ClassFilter()
        assert_all_pass(filter_obj, single_detection)

    def test_neither_parameter_with_empty_detection(self, empty_detection):
        """Test that empty detection works with no restrictions."""
        filter_obj = ClassFilter()
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestClassFilterEdgeCases:
    """Test edge cases and boundary conditions for ClassFilter."""

    def test_class_id_as_integer(self, multi_class_detection):
        """Test that class IDs are correctly converted to integers.

        The filter should handle numpy types and ensure proper comparison.
        """
        filter_obj = ClassFilter(allowed_classes=[1])
        # Verify that index 1 (class 1) passes
        is_valid = filter_obj.is_valid(
            multi_class_detection.image, 1, multi_class_detection
        )
        assert is_valid is True

    def test_allowed_classes_with_large_class_ids(self, sample_image):
        """Test filter with large class ID values."""
        import numpy as np

        from action_labeler.detections.detection import Detection

        # Create detection with large class IDs
        xyxy = np.array([[100, 100, 200, 200], [250, 250, 350, 350]])
        class_id = np.array([100, 200])
        segmentation_points = [
            [0.156, 0.208, 0.312, 0.208, 0.312, 0.417, 0.156, 0.417],
            [0.391, 0.521, 0.547, 0.521, 0.547, 0.729, 0.391, 0.729],
        ]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        filter_obj = ClassFilter(allowed_classes=[100])
        expected_valid_indices = [0]  # Only class 100
        assert_filter_validates_indices(filter_obj, detection, expected_valid_indices)

    def test_excluded_classes_with_duplicate_values(self):
        """Test that duplicate values in excluded_classes are handled correctly.

        Duplicates should be removed when converting to a set.
        """
        filter_obj = ClassFilter(excluded_classes=[0, 1, 1, 2, 2, 2])
        assert filter_obj.excluded_classes == {0, 1, 2}
        assert len(filter_obj.excluded_classes) == 3

    def test_filter_consistency_across_multiple_calls(self, multi_class_detection):
        """Test that filter returns consistent results across multiple calls.

        Calling is_valid multiple times for the same detection should
        return the same result.
        """
        filter_obj = ClassFilter(allowed_classes=[0, 1])

        # Call is_valid multiple times for each index
        for index in range(len(multi_class_detection.xyxy)):
            result1 = filter_obj.is_valid(
                multi_class_detection.image, index, multi_class_detection
            )
            result2 = filter_obj.is_valid(
                multi_class_detection.image, index, multi_class_detection
            )
            result3 = filter_obj.is_valid(
                multi_class_detection.image, index, multi_class_detection
            )
            assert result1 == result2 == result3


class TestClassFilterHelperFunctions:
    """Test ClassFilter using helper functions to verify integration."""

    def test_count_passing_detections(self, multi_class_detection):
        """Test counting how many detections pass the filter.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With allowed_classes=[0], should have 2 passing detections.
        """
        filter_obj = ClassFilter(allowed_classes=[0])
        count = count_passing_detections(filter_obj, multi_class_detection)
        assert count == 2

    def test_get_passing_indices(self, multi_class_detection):
        """Test getting list of indices that pass the filter.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With allowed_classes=[1], should return indices [1, 4].
        """
        filter_obj = ClassFilter(allowed_classes=[1])
        passing = get_passing_indices(filter_obj, multi_class_detection)
        assert passing == [1, 4]

    def test_get_failing_indices(self, multi_class_detection):
        """Test getting list of indices that fail the filter.

        multi_class_detection has classes: [0, 1, 2, 0, 1]
        With allowed_classes=[0, 1], should return index [2] (class 2).
        """
        filter_obj = ClassFilter(allowed_classes=[0, 1])
        failing = get_failing_indices(filter_obj, multi_class_detection)
        assert failing == [2]

    def test_assert_filter_validates_indices_helper(self, multi_class_detection):
        """Test that assert_filter_validates_indices works correctly with ClassFilter.

        This verifies the integration between ClassFilter and test helpers.
        """
        filter_obj = ClassFilter(excluded_classes=[2])
        expected_valid_indices = [0, 1, 3, 4]

        # This should not raise an assertion error
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

        # This should raise an assertion error (incorrect expectation)
        with pytest.raises(AssertionError):
            assert_filter_validates_indices(filter_obj, multi_class_detection, [0, 1, 2])
