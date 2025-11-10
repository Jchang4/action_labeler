"""Helper functions for filter testing.

This module provides utility functions for common test assertions
and operations when testing filters.
"""

from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter


def assert_filter_validates_indices(
    filter_obj: IFilter,
    detection: Detection,
    expected_valid_indices: list[int],
    image: Image.Image | None = None,
) -> None:
    """Assert that a filter validates exactly the expected indices.

    Args:
        filter_obj: The filter to test
        detection: Detection object to test against
        expected_valid_indices: List of indices that should pass the filter
        image: Optional image (uses detection.image if not provided)

    Raises:
        AssertionError: If filter results don't match expectations
    """
    if image is None:
        image = detection.image

    num_detections = len(detection.xyxy)
    expected_valid_set = set(expected_valid_indices)

    for i in range(num_detections):
        is_valid = filter_obj.is_valid(image, i, detection)
        should_be_valid = i in expected_valid_set

        assert is_valid == should_be_valid, (
            f"Filter validation failed for index {i}: "
            f"expected {should_be_valid}, got {is_valid}"
        )


def assert_all_pass(
    filter_obj: IFilter,
    detection: Detection,
    image: Image.Image | None = None,
) -> None:
    """Assert that all detections pass the filter.

    Args:
        filter_obj: The filter to test
        detection: Detection object to test against
        image: Optional image (uses detection.image if not provided)

    Raises:
        AssertionError: If any detection fails the filter
    """
    if image is None:
        image = detection.image

    num_detections = len(detection.xyxy)
    expected_valid_indices = list(range(num_detections))

    assert_filter_validates_indices(filter_obj, detection, expected_valid_indices, image)


def assert_all_fail(
    filter_obj: IFilter,
    detection: Detection,
    image: Image.Image | None = None,
) -> None:
    """Assert that all detections fail the filter.

    Args:
        filter_obj: The filter to test
        detection: Detection object to test against
        image: Optional image (uses detection.image if not provided)

    Raises:
        AssertionError: If any detection passes the filter
    """
    if image is None:
        image = detection.image

    assert_filter_validates_indices(filter_obj, detection, [], image)


def assert_none_pass(
    filter_obj: IFilter,
    detection: Detection,
    image: Image.Image | None = None,
) -> None:
    """Alias for assert_all_fail for better readability in some contexts.

    Args:
        filter_obj: The filter to test
        detection: Detection object to test against
        image: Optional image (uses detection.image if not provided)

    Raises:
        AssertionError: If any detection passes the filter
    """
    assert_all_fail(filter_obj, detection, image)


def count_passing_detections(
    filter_obj: IFilter,
    detection: Detection,
    image: Image.Image | None = None,
) -> int:
    """Count how many detections pass the filter.

    Args:
        filter_obj: The filter to test
        detection: Detection object to test against
        image: Optional image (uses detection.image if not provided)

    Returns:
        Number of detections that pass the filter
    """
    if image is None:
        image = detection.image

    count = 0
    for i in range(len(detection.xyxy)):
        if filter_obj.is_valid(image, i, detection):
            count += 1

    return count


def get_passing_indices(
    filter_obj: IFilter,
    detection: Detection,
    image: Image.Image | None = None,
) -> list[int]:
    """Get list of indices that pass the filter.

    Args:
        filter_obj: The filter to test
        detection: Detection object to test against
        image: Optional image (uses detection.image if not provided)

    Returns:
        List of indices that pass the filter
    """
    if image is None:
        image = detection.image

    passing = []
    for i in range(len(detection.xyxy)):
        if filter_obj.is_valid(image, i, detection):
            passing.append(i)

    return passing


def get_failing_indices(
    filter_obj: IFilter,
    detection: Detection,
    image: Image.Image | None = None,
) -> list[int]:
    """Get list of indices that fail the filter.

    Args:
        filter_obj: The filter to test
        detection: Detection object to test against
        image: Optional image (uses detection.image if not provided)

    Returns:
        List of indices that fail the filter
    """
    if image is None:
        image = detection.image

    failing = []
    for i in range(len(detection.xyxy)):
        if not filter_obj.is_valid(image, i, detection):
            failing.append(i)

    return failing
