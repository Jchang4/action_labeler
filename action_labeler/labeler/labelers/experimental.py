"""Experimental labeler for interactive research workflows.

This labeler is designed for experimentation and iteration:
- Process images one at a time
- Preview prompts before execution
- Visualize preprocessed images
- Immediate feedback and inspection
- Support both single and batch detection modes
"""

from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from action_labeler.labeler.core.experiment import ExperimentConfig
from action_labeler.labeler.core.image_provider import IImageProvider, ImageData
from action_labeler.labeler.core.processing_modes import (
    IProcessingMode,
    ProcessingUnit,
    get_processing_mode,
)
from action_labeler.labeler.core.processing_pipeline import ProcessingPipeline
from action_labeler.labeler.storage.label_store import LabelStore
from action_labeler.labeler.storage.metadata import LabeledDetection, LabelMetadata


class ExperimentalLabeler:
    """Interactive labeler for research and experimentation.

    Designed for:
    - Testing different prompts on single images
    - Visualizing preprocessing effects
    - Comparing single vs batch detection modes
    - Iterative refinement of labeling approach

    Key features:
    - Process one image at a time with immediate feedback
    - Preview prompts and preprocessed images before labeling
    - Support dry-run mode to see what would be labeled
    - Easy switching between processing modes
    - Rich metadata tracking for reproducibility
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        pipeline: ProcessingPipeline,
        image_provider: IImageProvider,
        processing_mode: IProcessingMode | None = None,
    ):
        """Initialize experimental labeler.

        Args:
            experiment: Experiment configuration
            pipeline: Processing pipeline (filters, preprocessors, model)
            image_provider: Source of images and detections
            processing_mode: How to process detections (auto-created from experiment if None)
        """
        self.experiment = experiment
        self.pipeline = pipeline
        self.image_provider = image_provider
        self.processing_mode = processing_mode or get_processing_mode(
            experiment.processing_mode
        )

        # Storage for labeled data
        self.label_store = LabelStore()

        # Track statistics
        self.stats = {
            "images_processed": 0,
            "detections_labeled": 0,
            "detections_filtered": 0,
            "invalid_labels": 0,
        }

    def label_image(
        self, image_data: ImageData, dry_run: bool = False, show_preview: bool = False
    ) -> list[LabeledDetection]:
        """Label a single image interactively.

        Args:
            image_data: Image and detections to label
            dry_run: If True, don't actually call model or save results
            show_preview: If True, show preprocessed images (requires matplotlib)

        Returns:
            List of labeled detections
        """
        # Create processing units based on mode
        units = self.processing_mode.create_processing_units(
            image_data.image, image_data.image_path, image_data.detections
        )

        labeled_detections = []

        for unit in units:
            # Skip if already labeled
            if self._is_already_labeled(unit):
                continue

            # Check filters
            if not self.pipeline.should_process(unit):
                self.stats["detections_filtered"] += 1
                continue

            # Preprocess
            preprocessed_image = self.pipeline.preprocess(unit)

            # Show preview if requested
            if show_preview:
                self._show_preview(preprocessed_image, unit, image_data.image_path)

            # Generate prompt
            prompt = self.pipeline.generate_prompt(unit)

            if dry_run:
                print(f"[DRY RUN] Would label detection with prompt:")
                print(f"  {prompt}")
                continue

            # Query model
            response = self.pipeline.process(unit)

            if response is None:
                # Filtered out
                self.stats["detections_filtered"] += 1
                continue

            # Track invalid labels
            if not response.is_valid:
                self.stats["invalid_labels"] += 1

            # Create labeled detection
            labeled_detection = self._create_labeled_detection(
                unit, response, image_data.image_path
            )

            # Add to store
            self.label_store.add(labeled_detection)
            labeled_detections.append(labeled_detection)

            self.stats["detections_labeled"] += 1

        self.stats["images_processed"] += 1
        return labeled_detections

    def label_all(
        self,
        max_images: int | None = None,
        show_progress: bool = True,
        checkpoint_every: int = 10,
        checkpoint_path: str | Path | None = None,
    ) -> LabelStore:
        """Label all images from the provider.

        Args:
            max_images: Maximum number of images to process (None = all)
            show_progress: Whether to show progress bar
            checkpoint_every: Save checkpoint every N images
            checkpoint_path: Path to save checkpoints (default: experiment_name_checkpoint.pkl)

        Returns:
            LabelStore with all labeled detections
        """
        if checkpoint_path is None:
            checkpoint_path = f"{self.experiment.name}_checkpoint.pkl"

        images_processed = 0

        # Create progress bar
        iterator = self.image_provider
        if show_progress:
            total = (
                min(max_images, len(self.image_provider))
                if max_images
                else len(self.image_provider)
            )
            iterator = tqdm(
                iterator,
                total=total,
                desc=f"Labeling ({self.experiment.name})",
            )

        for image_data in iterator:
            # Label this image
            self.label_image(image_data)

            images_processed += 1

            # Checkpoint
            if checkpoint_every and images_processed % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path)

            # Check max limit
            if max_images and images_processed >= max_images:
                break

        # Final save
        self.label_store.flush()
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path)

        return self.label_store

    def preview_prompt(self, image_data: ImageData, detection_index: int = 0) -> str:
        """Preview the prompt that would be used for a detection.

        Useful for testing prompts before running expensive model calls.

        Args:
            image_data: Image and detections
            detection_index: Which detection to preview (for single mode)

        Returns:
            Generated prompt string
        """
        units = self.processing_mode.create_processing_units(
            image_data.image, image_data.image_path, image_data.detections
        )

        if detection_index >= len(units):
            raise IndexError(
                f"Detection index {detection_index} out of range (0-{len(units)-1})"
            )

        unit = units[detection_index]
        return self.pipeline.generate_prompt(unit)

    def preview_preprocessing(
        self, image_data: ImageData, detection_index: int = 0
    ) -> Image.Image:
        """Preview the preprocessed image for a detection.

        Args:
            image_data: Image and detections
            detection_index: Which detection to preview

        Returns:
            Preprocessed PIL Image
        """
        units = self.processing_mode.create_processing_units(
            image_data.image, image_data.image_path, image_data.detections
        )

        if detection_index >= len(units):
            raise IndexError(
                f"Detection index {detection_index} out of range (0-{len(units)-1})"
            )

        unit = units[detection_index]
        return self.pipeline.preprocess(unit)

    def get_stats(self) -> dict[str, Any]:
        """Get labeling statistics.

        Returns:
            Dictionary with statistics
        """
        store_stats = self.label_store.get_statistics()

        return {
            **self.stats,
            **store_stats,
            "experiment": self.experiment.name,
            "processing_mode": self.processing_mode.get_name(),
        }

    def print_stats(self) -> None:
        """Print labeling statistics to console."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print(f"Experiment: {stats['experiment']}")
        print(f"Processing Mode: {stats['processing_mode']}")
        print("=" * 50)
        print(f"Images Processed: {stats['images_processed']}")
        print(f"Detections Labeled: {stats['detections_labeled']}")
        print(f"Detections Filtered: {stats['detections_filtered']}")
        print(f"Invalid Labels: {stats['invalid_labels']}")
        print(f"Unique Labels: {stats['unique_labels']}")
        print(f"Unique Images: {stats['unique_images']}")
        print("=" * 50)

        if stats.get("label_distribution"):
            print("\nLabel Distribution:")
            for label, count in sorted(
                stats["label_distribution"].items(), key=lambda x: -x[1]
            ):
                print(f"  {label}: {count}")
        print()

    def _is_already_labeled(self, unit: ProcessingUnit) -> bool:
        """Check if a processing unit is already labeled.

        Args:
            unit: Processing unit to check

        Returns:
            True if already exists in store
        """
        image_path = unit.metadata.get("image_path", "") if unit.metadata else ""

        # For batch mode, check if any detection from this image is labeled
        if unit.detection_index is None:
            # Check all detections in this image
            for i in range(len(unit.detection.xyxy)):
                xywh = list(unit.detection.xywh[i])
                if self.label_store.exists(image_path, xywh):
                    return True
            return False
        else:
            # Single detection mode
            xywh = list(unit.detection.xywh[unit.detection_index])
            return self.label_store.exists(image_path, xywh)

    def _create_labeled_detection(
        self, unit: ProcessingUnit, response: Any, image_path: str
    ) -> LabeledDetection:
        """Create a LabeledDetection from processing results.

        Args:
            unit: Processing unit
            response: Model response
            image_path: Path to source image

        Returns:
            LabeledDetection object
        """
        # Get detection coordinates
        detection_idx = unit.detection_index if unit.detection_index is not None else 0
        xywh = list(unit.detection.xywh[detection_idx])

        # Get segmentation points (if available)
        seg_points = []
        if unit.detection.segmentation_points and detection_idx < len(
            unit.detection.segmentation_points
        ):
            seg_points = unit.detection.segmentation_points[detection_idx]

        # Create metadata
        metadata = LabelMetadata(
            experiment_id=self.experiment.get_hash(),
            model_name=self.experiment.model_name,
            prompt_version=self.experiment.prompt_version,
            processing_mode=self.processing_mode.get_name(),
            confidence=response.confidence,
            raw_model_response=response.raw_response,
            is_valid=response.is_valid,
            validation_error=response.validation_error,
            preprocessors_applied=[
                p.__class__.__name__ for p in self.pipeline.preprocessors
            ],
            filters_applied=[f.__class__.__name__ for f in self.pipeline.filters],
        )

        return LabeledDetection(
            image_path=image_path,
            xywh=xywh,
            segmentation_points=seg_points,
            label=response.label,
            metadata=metadata,
        )

    def _save_checkpoint(self, path: str | Path) -> None:
        """Save checkpoint to disk.

        Args:
            path: Path to save checkpoint
        """
        from action_labeler.labeler.storage.persistence import LabelPersistence

        LabelPersistence.save(self.label_store, path)

    def _show_preview(
        self, image: Image.Image, unit: ProcessingUnit, image_path: str
    ) -> None:
        """Show preview of preprocessed image.

        Args:
            image: Preprocessed image
            unit: Processing unit
            image_path: Path to source image
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis("off")

            # Add title
            mode = self.processing_mode.get_name()
            det_idx = (
                unit.detection_index if unit.detection_index is not None else "all"
            )
            plt.title(f"{Path(image_path).name} | Mode: {mode} | Detection: {det_idx}")

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Warning: matplotlib not available for preview")
