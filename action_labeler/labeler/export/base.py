"""Base interface for dataset exporters.

Defines the contract for exporting labeled data to various formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from action_labeler.labeler.storage.label_store import LabelStore


class IDatasetExporter(ABC):
    """Interface for exporting labeled datasets to specific formats.

    Exporters convert LabelStore data into format-specific outputs
    (YOLO, COCO, custom formats, etc.)
    """

    @abstractmethod
    def export(
        self,
        label_store: LabelStore,
        output_path: str | Path,
        **kwargs,
    ) -> None:
        """Export label store to the target format.

        Args:
            label_store: Store containing labeled detections
            output_path: Path to output location (file or directory)
            **kwargs: Format-specific export options
        """
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """Get the name of this export format.

        Returns:
            Format name (e.g., "yolov8", "coco", "csv")
        """
        pass

    @abstractmethod
    def validate_export(self, label_store: LabelStore) -> tuple[bool, list[str]]:
        """Validate that the label store can be exported to this format.

        Args:
            label_store: Store to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
