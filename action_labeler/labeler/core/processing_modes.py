"""Processing modes for different labeling strategies.

This module defines how images and detections are processed:
- SingleDetectionMode: Process each detection independently (current behavior)
- BatchDetectionMode: Process all detections in an image together (new)
- HybridMode: Image-level context followed by detection-level labels (new)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image

from action_labeler.detections.detection import Detection


@dataclass
class ProcessingUnit:
    """A unit of work to be processed by the labeling pipeline.

    Attributes:
        image: The image to process
        detection: Detection(s) to label
        detection_index: Index of specific detection (for single mode)
        metadata: Additional metadata for this unit
    """

    image: Image.Image
    detection: Detection
    detection_index: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class LabelResult:
    """Result from labeling a processing unit.

    Attributes:
        image_path: Path to the source image
        xywh: Bounding box coordinates (normalized)
        segmentation_points: Segmentation polygon points
        label: The assigned label/action
        confidence: Optional confidence score
        raw_response: Raw model response
        metadata: Additional metadata
    """

    image_path: str
    xywh: list[float]
    segmentation_points: list[list[float]]
    label: str
    confidence: float | None = None
    raw_response: str = ""
    metadata: dict[str, Any] | None = None


class IProcessingMode(ABC):
    """Interface for different processing modes.

    Processing modes determine how images and detections are batched
    and sent to the labeling pipeline.
    """

    @abstractmethod
    def create_processing_units(
        self, image: Image.Image, image_path: str, detections: Detection
    ) -> list[ProcessingUnit]:
        """Create processing units from an image and its detections.

        Args:
            image: The source image
            image_path: Path to the image
            detections: All detections in the image

        Returns:
            List of processing units to be labeled
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this processing mode."""
        pass

    @abstractmethod
    def requires_preprocessing(self) -> bool:
        """Whether this mode requires preprocessing (e.g., cropping).

        Returns:
            True if preprocessors should be applied per detection
        """
        pass


class SingleDetectionMode(IProcessingMode):
    """Process each detection independently.

    This is the traditional mode where each detection is cropped/masked
    and labeled individually without context from other detections.

    Best for:
    - Simple object classification
    - When detections are independent
    - When memory is limited (one detection at a time)
    """

    def create_processing_units(
        self, image: Image.Image, image_path: str, detections: Detection
    ) -> list[ProcessingUnit]:
        """Create one processing unit per detection.

        Each unit contains the full detection set but specifies which
        detection index to focus on (for preprocessing).

        Args:
            image: The source image
            image_path: Path to the image
            detections: All detections in the image

        Returns:
            One ProcessingUnit per detection
        """
        units = []

        for i in range(len(detections.xyxy)):
            unit = ProcessingUnit(
                image=image.copy(),
                detection=detections,
                detection_index=i,
                metadata={"image_path": image_path, "mode": "single"},
            )
            units.append(unit)

        return units

    def get_name(self) -> str:
        """Get the name of this processing mode."""
        return "single"

    def requires_preprocessing(self) -> bool:
        """Single detection mode typically uses preprocessing (crop/mask)."""
        return True


class BatchDetectionMode(IProcessingMode):
    """Process all detections in an image together.

    Sends the entire image with all detections to the model at once,
    asking it to label all detections together. This provides context
    and can capture relationships between detections.

    Best for:
    - Actions that require scene context (e.g., "cooking in kitchen")
    - Multi-person scenes with interactions
    - When relationships between objects matter
    - Reducing API calls (one call per image vs per detection)
    """

    def create_processing_units(
        self, image: Image.Image, image_path: str, detections: Detection
    ) -> list[ProcessingUnit]:
        """Create a single processing unit for all detections.

        Args:
            image: The source image
            image_path: Path to the image
            detections: All detections in the image

        Returns:
            Single ProcessingUnit containing all detections
        """
        unit = ProcessingUnit(
            image=image,
            detection=detections,
            detection_index=None,  # Process all detections
            metadata={
                "image_path": image_path,
                "mode": "batch",
                "num_detections": len(detections.xyxy),
            },
        )

        return [unit]

    def get_name(self) -> str:
        """Get the name of this processing mode."""
        return "batch"

    def requires_preprocessing(self) -> bool:
        """Batch mode typically doesn't crop (uses full image context)."""
        return False


class HybridMode(IProcessingMode):
    """Two-stage processing: image-level context + detection-level labels.

    First stage: Analyze the entire image to understand scene context
    Second stage: Label each detection using the scene context

    This combines the benefits of both modes:
    - Scene understanding from batch mode
    - Detailed per-detection labels from single mode

    Best for:
    - Complex scenes requiring context
    - When you want both scene description and detection labels
    - Research comparing context-aware vs context-free labeling
    """

    def __init__(self, include_scene_context: bool = True):
        """Initialize hybrid mode.

        Args:
            include_scene_context: If True, first stage analyzes scene.
                                  If False, just marks for two-pass processing.
        """
        self.include_scene_context = include_scene_context

    def create_processing_units(
        self, image: Image.Image, image_path: str, detections: Detection
    ) -> list[ProcessingUnit]:
        """Create processing units for two-stage labeling.

        Creates one unit for scene context (if enabled) plus one per detection.

        Args:
            image: The source image
            image_path: Path to the image
            detections: All detections in the image

        Returns:
            List of processing units (scene + individual detections)
        """
        units = []

        # Stage 1: Scene context (optional)
        if self.include_scene_context:
            scene_unit = ProcessingUnit(
                image=image.copy(),
                detection=detections,
                detection_index=None,
                metadata={
                    "image_path": image_path,
                    "mode": "hybrid",
                    "stage": "scene_context",
                },
            )
            units.append(scene_unit)

        # Stage 2: Individual detections
        for i in range(len(detections.xyxy)):
            detection_unit = ProcessingUnit(
                image=image.copy(),
                detection=detections,
                detection_index=i,
                metadata={
                    "image_path": image_path,
                    "mode": "hybrid",
                    "stage": "detection",
                    "detection_index": i,
                },
            )
            units.append(detection_unit)

        return units

    def get_name(self) -> str:
        """Get the name of this processing mode."""
        return "hybrid"

    def requires_preprocessing(self) -> bool:
        """Hybrid mode uses preprocessing for detection stage."""
        return True


def get_processing_mode(mode_name: str, **kwargs: Any) -> IProcessingMode:
    """Factory function to create processing mode by name.

    Args:
        mode_name: Name of mode ("single", "batch", or "hybrid")
        **kwargs: Additional arguments for mode initialization

    Returns:
        Processing mode instance

    Raises:
        ValueError: If mode_name is invalid
    """
    if mode_name == "single":
        return SingleDetectionMode()
    elif mode_name == "batch":
        return BatchDetectionMode()
    elif mode_name == "hybrid":
        return HybridMode(**kwargs)
    else:
        raise ValueError(
            f"Invalid processing mode: {mode_name}. "
            "Must be 'single', 'batch', or 'hybrid'"
        )
