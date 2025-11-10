"""Metadata tracking for labeled detections.

This module provides rich metadata storage for each labeled detection,
enabling experiment tracking and reproducibility.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LabelMetadata:
    """Metadata for a single labeled detection.

    Tracks all information needed to reproduce and understand a label:
    - What model produced it
    - What prompt was used
    - When it was created
    - Confidence/quality indicators
    - Processing parameters

    Attributes:
        experiment_id: ID of experiment configuration
        model_name: Name of VLM used
        prompt_version: Version of prompt template
        timestamp: When label was created
        confidence: Model confidence score (0-1)
        processing_mode: "single", "batch", or "hybrid"
        preprocessors_applied: List of preprocessor names
        filters_applied: List of filter names
        raw_model_response: Complete model output
        is_valid: Whether label passed validation
        validation_error: Error if validation failed
        custom_metadata: Additional custom fields
    """

    experiment_id: str
    model_name: str
    prompt_version: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float | None = None
    processing_mode: str = "single"
    preprocessors_applied: list[str] = field(default_factory=list)
    filters_applied: list[str] = field(default_factory=list)
    raw_model_response: str = ""
    is_valid: bool = True
    validation_error: str | None = None
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabelMetadata":
        """Create from dictionary."""
        return cls(**data)

    def to_compact_dict(self) -> dict[str, Any]:
        """Convert to compact dictionary (exclude large fields).

        Useful for displaying summaries without full raw responses.

        Returns:
            Dictionary with large fields excluded
        """
        data = self.to_dict()
        # Remove potentially large fields
        data.pop("raw_model_response", None)
        data.pop("custom_metadata", None)
        return data


@dataclass
class LabeledDetection:
    """A complete labeled detection with metadata.

    Combines the detection information with its label and metadata.

    Attributes:
        image_path: Path to source image
        xywh: Bounding box in normalized xywh format
        segmentation_points: Segmentation polygon points (empty for bbox)
        label: The assigned label/action
        metadata: Rich metadata about this label
    """

    image_path: str
    xywh: list[float]
    segmentation_points: list[list[float]]
    label: str
    metadata: LabelMetadata

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        # Handle case where metadata is already a dict (from DataFrame)
        metadata_dict = (
            self.metadata.to_dict()
            if isinstance(self.metadata, LabelMetadata)
            else self.metadata
        )

        return {
            "image_path": self.image_path,
            "xywh": self.xywh,
            "segmentation_points": self.segmentation_points,
            "label": self.label,
            "metadata": metadata_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabeledDetection":
        """Create from dictionary."""
        # Convert metadata dict to LabelMetadata object
        metadata_dict = data.get("metadata", {})
        metadata = LabelMetadata.from_dict(metadata_dict)

        return cls(
            image_path=data["image_path"],
            xywh=data["xywh"],
            segmentation_points=data["segmentation_points"],
            label=data["label"],
            metadata=metadata,
        )

    def matches_detection(self, image_path: str, xywh: list[float]) -> bool:
        """Check if this label matches a specific detection.

        Args:
            image_path: Image path to match
            xywh: Bounding box to match

        Returns:
            True if image path and xywh match
        """
        return self.image_path == image_path and self.xywh == xywh

    def get_detection_key(self) -> tuple[str, tuple[float, ...]]:
        """Get a hashable key for this detection.

        Returns:
            Tuple of (image_path, xywh_tuple)
        """
        return (self.image_path, tuple(self.xywh))


def create_metadata_from_experiment(
    experiment_id: str,
    model_name: str,
    prompt_version: str,
    processing_mode: str,
    raw_response: str,
    confidence: float | None = None,
    preprocessors: list[str] | None = None,
    filters: list[str] | None = None,
    is_valid: bool = True,
    validation_error: str | None = None,
) -> LabelMetadata:
    """Factory function to create metadata from experiment parameters.

    Args:
        experiment_id: ID of experiment
        model_name: Name of model
        prompt_version: Version of prompt
        processing_mode: Processing mode used
        raw_response: Raw model output
        confidence: Optional confidence score
        preprocessors: List of preprocessor names applied
        filters: List of filter names applied
        is_valid: Whether label is valid
        validation_error: Validation error if any

    Returns:
        LabelMetadata instance
    """
    return LabelMetadata(
        experiment_id=experiment_id,
        model_name=model_name,
        prompt_version=prompt_version,
        processing_mode=processing_mode,
        raw_model_response=raw_response,
        confidence=confidence,
        preprocessors_applied=preprocessors or [],
        filters_applied=filters or [],
        is_valid=is_valid,
        validation_error=validation_error,
    )
