from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter


class ClassFilter(IFilter):
    """Filter detections by class ID.

    Allows filtering to include only specific classes or exclude certain classes.
    Useful for focusing on particular object types or removing unwanted classes.

    Args:
        allowed_classes: List of class IDs to include (None means all allowed)
        excluded_classes: List of class IDs to exclude (None means none excluded)

    Note:
        If both allowed_classes and excluded_classes are specified,
        allowed_classes takes precedence (exclude is ignored).
    """

    allowed_classes: set[int] | None
    excluded_classes: set[int] | None

    def __init__(
        self,
        allowed_classes: list[int] | None = None,
        excluded_classes: list[int] | None = None,
    ):
        if allowed_classes is not None and excluded_classes is not None:
            raise ValueError(
                "Cannot specify both allowed_classes and excluded_classes. "
                "Use one or the other."
            )

        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        self.excluded_classes = set(excluded_classes) if excluded_classes else None

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        class_id = int(detections.class_id[index])

        # If allowed_classes is specified, class must be in the list
        if self.allowed_classes is not None:
            return class_id in self.allowed_classes

        # If excluded_classes is specified, class must NOT be in the list
        if self.excluded_classes is not None:
            return class_id not in self.excluded_classes

        # If neither is specified, allow all classes
        return True
