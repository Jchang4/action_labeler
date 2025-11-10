"""Custom exceptions for dataset operations.

This module defines custom exception classes for handling dataset-specific errors
with more context and better error messages.
"""


class DatasetError(Exception):
    """Base exception for all dataset-related errors."""

    pass


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails.

    This includes issues like:
    - Invalid coordinate ranges
    - Missing image files
    - Invalid class IDs
    - Corrupted data
    """

    pass


class ClassNotFoundError(DatasetError):
    """Raised when attempting to operate on a class that doesn't exist in the dataset."""

    def __init__(self, class_name: str, available_classes: list[str]):
        self.class_name = class_name
        self.available_classes = available_classes
        super().__init__(
            f"Class '{class_name}' not found in dataset. "
            f"Available classes: {', '.join(available_classes)}"
        )


class ClassMappingError(DatasetError):
    """Raised when class mapping/remapping fails."""

    pass


class DatasetIOError(DatasetError):
    """Raised when dataset I/O operations fail."""

    pass


class EmptyDatasetError(DatasetError):
    """Raised when attempting operations on an empty dataset that require data."""

    pass


class InvalidSplitError(DatasetError):
    """Raised when an invalid dataset split is specified."""

    def __init__(self, split: str, valid_splits: list[str]):
        self.split = split
        self.valid_splits = valid_splits
        super().__init__(
            f"Invalid split '{split}'. Valid splits are: {', '.join(valid_splits)}"
        )
