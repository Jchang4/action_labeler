"""Production labeler for efficient batch processing.

This labeler is optimized for processing large datasets:
- Efficient batch processing
- Automatic checkpointing with auto-resume
- Progress tracking and statistics
- Graceful error handling (continues on errors)
- Resume from interruptions (automatic by default)
- Optimized for throughput over interactivity
"""

from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from action_labeler.labeler.core.experiment import ExperimentConfig, ExperimentRun
from action_labeler.labeler.core.image_provider import IImageProvider, ImageData
from action_labeler.labeler.core.processing_modes import (
    IProcessingMode,
    ProcessingUnit,
    get_processing_mode,
)
from action_labeler.labeler.core.processing_pipeline import (
    ModelResponse,
    ProcessingPipeline,
)
from action_labeler.labeler.storage.label_store import LabelStore
from action_labeler.labeler.storage.metadata import LabeledDetection, LabelMetadata
from action_labeler.labeler.storage.persistence import LabelPersistence


class ProductionLabeler:
    """Production labeler for efficient batch dataset labeling.

    Designed for:
    - Processing large datasets efficiently
    - Automatic checkpointing and recovery
    - Detailed progress tracking
    - Production-ready reliability

    Key features:
    - Automatic checkpointing every N images
    - Auto-resume from latest checkpoint (default behavior)
    - Graceful error handling (continues on errors)
    - Detailed statistics tracking
    - Efficient batch processing
    - Experiment run tracking with unique IDs
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        pipeline: ProcessingPipeline,
        image_provider: IImageProvider,
        checkpoint_dir: str | Path = "./checkpoints",
        processing_mode: IProcessingMode | None = None,
    ):
        """Initialize production labeler.

        Args:
            experiment: Experiment configuration
            pipeline: Processing pipeline
            image_provider: Source of images and detections
            checkpoint_dir: Directory for checkpoints
            processing_mode: How to process detections (auto-created if None)
        """
        self.experiment = experiment
        self.pipeline = pipeline
        self.image_provider = image_provider
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.processing_mode = processing_mode or get_processing_mode(
            experiment.processing_mode
        )

        # Storage for labeled data
        self.label_store = LabelStore()

        # Track detailed statistics
        self.stats = {
            "images_processed": 0,
            "images_skipped": 0,
            "detections_labeled": 0,
            "detections_filtered": 0,
            "detections_skipped": 0,  # Already labeled
            "invalid_labels": 0,
            "errors": 0,
        }

        # Experiment run tracking
        self.run_id = self._generate_run_id()
        self.experiment_run: ExperimentRun | None = None

    def label_dataset(
        self,
        checkpoint_every: int = 50,
        save_final: bool = True,
        output_path: str | Path | None = None,
        resume_from: str | Path | None = None,
        auto_resume: bool = True,
        overwrite: bool = False,
    ) -> LabelStore:
        """Label entire dataset with automatic checkpoint resumption.

        By default, automatically resumes from the latest checkpoint if found,
        making it safe to interrupt and restart labeling runs without losing progress.

        Args:
            checkpoint_every: Save checkpoint every N images
            save_final: Whether to save final results
            output_path: Path for final results (default: experiment_name.pkl)
            resume_from: Explicit path to checkpoint to resume from (overrides auto_resume)
            auto_resume: If True, automatically resume from latest checkpoint (default: True)
            overwrite: If True, clear all checkpoints and start fresh (default: False)

        Returns:
            LabelStore with all labeled detections

        Example:
            >>> # Simple usage - auto-resumes from latest checkpoint
            >>> labeler.label_dataset()
            >>>
            >>> # Force fresh start (clear all checkpoints)
            >>> labeler.label_dataset(overwrite=True)
            >>>
            >>> # Resume from specific checkpoint
            >>> labeler.label_dataset(resume_from="checkpoint_500.pkl")
            >>>
            >>> # Disable auto-resume
            >>> labeler.label_dataset(auto_resume=False)
        """
        # Setup output path
        if output_path is None:
            output_path = self.checkpoint_dir / f"{self.experiment.name}.pkl"
        output_path = Path(output_path)

        # Handle checkpoint resumption with priority: overwrite > resume_from > auto_resume
        if overwrite:
            # Clear all existing checkpoints and start fresh
            self._clear_checkpoints()
            print("🆕 Starting fresh (all checkpoints cleared)")
        elif resume_from:
            # Explicit checkpoint specified - use it
            print(f"🔄 Resuming from specified checkpoint: {resume_from}")
            self._resume_from_checkpoint(resume_from)
        elif auto_resume:
            # Auto-detect and resume from latest checkpoint
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                print(
                    f"🔄 Auto-resuming from latest checkpoint: {latest_checkpoint.name}"
                )
                self._resume_from_checkpoint(latest_checkpoint)
            else:
                print("🆕 No checkpoints found, starting fresh")
        else:
            # auto_resume=False and no explicit checkpoint - start fresh
            print("🆕 Starting fresh (auto_resume disabled)")

        # Initialize experiment run
        self.experiment_run = ExperimentRun(
            experiment_config=self.experiment,
            run_id=self.run_id,
            results_path=output_path,
        )

        try:
            # Process all images
            self._process_all_images(checkpoint_every)

            # Mark run as completed
            if self.experiment_run:
                self.experiment_run.mark_completed(
                    num_images=self.stats["images_processed"],
                    num_labels=self.stats["detections_labeled"],
                )

            # Final save
            if save_final:
                self._save_results(output_path)

        except Exception as e:
            # Mark run as failed
            if self.experiment_run:
                self.experiment_run.mark_failed(str(e))
            raise

        return self.label_store

    def _process_all_images(self, checkpoint_every: int) -> None:
        """Process all images with progress tracking.

        Args:
            checkpoint_every: Checkpoint frequency
        """
        total_images = len(self.image_provider)

        # Create progress bar
        with tqdm(
            total=total_images,
            desc=f"Labeling dataset ({self.experiment.name})",
            unit="images",
        ) as pbar:
            for image_data in self.image_provider:
                try:
                    # Process this image
                    labeled_count = self._process_image(image_data)

                    self.stats["images_processed"] += 1

                    # Update progress bar with detailed stats
                    pbar.set_postfix(
                        {
                            "labeled": self.stats["detections_labeled"],
                            "filtered": self.stats["detections_filtered"],
                            "skipped": self.stats["detections_skipped"],
                            "errors": self.stats["errors"],
                        }
                    )
                    pbar.update(1)

                    # Checkpoint
                    if (
                        checkpoint_every
                        and self.stats["images_processed"] % checkpoint_every == 0
                    ):
                        self._save_checkpoint()

                except Exception as e:
                    # Log error and continue
                    self.stats["errors"] += 1
                    self.stats["images_skipped"] += 1
                    print(f"\nError processing {image_data.image_path}: {e}")
                    pbar.update(1)

        # Final flush
        self.label_store.flush()

    def _process_image(self, image_data: ImageData) -> int:
        """Process a single image and its detections.

        Args:
            image_data: Image and detections to process

        Returns:
            Number of detections labeled
        """
        labeled_count = 0

        # Create processing units
        units = self.processing_mode.create_processing_units(
            image_data.image, image_data.image_path, image_data.detections
        )

        for unit in units:
            # Skip if already labeled
            if self._is_already_labeled(unit):
                self.stats["detections_skipped"] += 1
                continue

            # Process through pipeline
            response = self.pipeline.process(unit)

            if response is None:
                # Filtered out
                self.stats["detections_filtered"] += 1
                continue

            # Track invalid labels
            if not response.is_valid:
                self.stats["invalid_labels"] += 1

            # Check if this is a batch response (multiple detections)
            is_batch = response.metadata.get("batch_mode", False)

            if is_batch:
                # Handle batch response: split into individual detections
                labeled_count += self._process_batch_response(
                    unit, response, image_data.image_path
                )
            else:
                # Handle single detection response
                labeled_detection = self._create_labeled_detection(
                    unit, response, image_data.image_path
                )
                self.label_store.add(labeled_detection)
                labeled_count += 1
                self.stats["detections_labeled"] += 1

        return labeled_count

    def _process_batch_response(
        self, unit: ProcessingUnit, response: ModelResponse, image_path: str
    ) -> int:
        """Process a batch response and create individual labeled detections.

        Args:
            unit: Processing unit (contains all detections)
            response: Model response with batch_labels in metadata
            image_path: Path to source image

        Returns:
            Number of detections labeled
        """
        labeled_count = 0

        # Extract batch labels from response metadata
        batch_labels = response.metadata.get("batch_labels", {})

        if not batch_labels:
            # No labels extracted (parsing may have failed)
            return 0

        # Create a labeled detection for each detection in the batch
        for detection_idx, label in batch_labels.items():
            # Get detection coordinates
            xywh = list(unit.detection.xywh[detection_idx])

            # Get segmentation points (if available)
            seg_points = []
            if unit.detection.segmentation_points and detection_idx < len(
                unit.detection.segmentation_points
            ):
                seg_points = unit.detection.segmentation_points[detection_idx]

            # Create metadata for this detection
            metadata = LabelMetadata(
                experiment_id=self.experiment.get_hash(),
                model_name=self.experiment.model_name,
                prompt_version=self.experiment.prompt_version,
                processing_mode=self.processing_mode.get_name(),
                confidence=response.confidence,
                raw_model_response=response.raw_response,  # Same for all detections
                is_valid=response.is_valid,
                validation_error=response.validation_error,
                preprocessors_applied=[
                    p.__class__.__name__ for p in self.pipeline.preprocessors
                ],
                filters_applied=[f.__class__.__name__ for f in self.pipeline.filters],
            )

            # Create labeled detection
            labeled_detection = LabeledDetection(
                image_path=image_path,
                xywh=xywh,
                segmentation_points=seg_points,
                label=label,  # Individual label for this detection
                metadata=metadata,
            )

            # Add to store
            self.label_store.add(labeled_detection)
            labeled_count += 1
            self.stats["detections_labeled"] += 1

        return labeled_count

    def _is_already_labeled(self, unit: ProcessingUnit) -> bool:
        """Check if processing unit is already labeled.

        Args:
            unit: Processing unit to check

        Returns:
            True if already exists in store
        """
        image_path = unit.metadata.get("image_path", "") if unit.metadata else ""

        if unit.detection_index is None:
            # Batch mode: check if any detection is labeled
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
        self, unit: ProcessingUnit, response: ModelResponse, image_path: str
    ) -> LabeledDetection:
        """Create LabeledDetection from processing results.

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

        # Get segmentation points
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

    def _save_checkpoint(self) -> None:
        """Save checkpoint with current state."""
        checkpoint_path = (
            self.checkpoint_dir
            / f"{self.experiment.name}_checkpoint_{self.stats['images_processed']}.pkl"
        )
        LabelPersistence.save(self.label_store, checkpoint_path)

        # Also save stats
        stats_path = checkpoint_path.with_suffix(".stats.json")
        import json

        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

    def _save_results(self, path: Path) -> None:
        """Save final results.

        Args:
            path: Path to save results
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        LabelPersistence.save(self.label_store, path)

        # Also save experiment run info
        if self.experiment_run:
            run_path = path.with_suffix(".run.json")
            import json

            with open(run_path, "w") as f:
                json.dump(self.experiment_run.to_dict(), f, indent=2)

    def _resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume from a previous checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load label store
        self.label_store = LabelPersistence.load(checkpoint_path)

        # Load stats if available
        stats_path = checkpoint_path.with_suffix(".stats.json")
        if stats_path.exists():
            import json

            with open(stats_path, "r") as f:
                self.stats = json.load(f)

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"Previously processed: {self.stats['images_processed']} images")
        print(f"Previously labeled: {self.stats['detections_labeled']} detections")

    def _find_latest_checkpoint(self) -> Path | None:
        """Find the most recent checkpoint in checkpoint_dir.

        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        pattern = f"{self.experiment.name}_checkpoint_*.pkl"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        if not checkpoints:
            return None

        # Sort by the numeric suffix (images_processed count)
        def get_checkpoint_number(path: Path) -> int:
            try:
                # Extract number from "experiment_checkpoint_100.pkl"
                return int(path.stem.split("_")[-1])
            except (ValueError, IndexError):
                return 0

        checkpoints.sort(key=get_checkpoint_number)
        return checkpoints[-1]  # Return the latest

    def _clear_checkpoints(self) -> None:
        """Delete all checkpoints for this experiment."""
        pattern = f"{self.experiment.name}_checkpoint_*"
        deleted_count = 0

        for path in self.checkpoint_dir.glob(pattern):
            path.unlink()
            deleted_count += 1

        if deleted_count > 0:
            print(f"   🗑️  Deleted {deleted_count} checkpoint file(s)")

    def _generate_run_id(self) -> str:
        """Generate unique run ID.

        Returns:
            Run ID string
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment.name}_{timestamp}"

    def get_stats(self) -> dict[str, Any]:
        """Get detailed statistics.

        Returns:
            Dictionary with statistics
        """
        store_stats = self.label_store.get_statistics()

        return {
            **self.stats,
            **store_stats,
            "experiment": self.experiment.name,
            "processing_mode": self.processing_mode.get_name(),
            "run_id": self.run_id,
        }

    def print_stats(self) -> None:
        """Print detailed statistics to console."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print(f"Production Run: {stats['run_id']}")
        print(f"Experiment: {stats['experiment']}")
        print(f"Processing Mode: {stats['processing_mode']}")
        print("=" * 60)
        print(f"Images Processed: {stats['images_processed']}")
        print(f"Images Skipped: {stats['images_skipped']}")
        print(f"Errors: {stats['errors']}")
        print("-" * 60)
        print(f"Detections Labeled: {stats['detections_labeled']}")
        print(f"Detections Filtered: {stats['detections_filtered']}")
        print(f"Detections Skipped (already labeled): {stats['detections_skipped']}")
        print(f"Invalid Labels: {stats['invalid_labels']}")
        print("-" * 60)
        print(f"Unique Labels: {stats['unique_labels']}")
        print(f"Unique Images: {stats['unique_images']}")
        print("=" * 60)

        if stats.get("label_distribution"):
            print("\nLabel Distribution:")
            for label, count in sorted(
                stats["label_distribution"].items(), key=lambda x: -x[1]
            )[
                :10
            ]:  # Top 10
                print(f"  {label}: {count}")
            if len(stats["label_distribution"]) > 10:
                print(f"  ... and {len(stats['label_distribution']) - 10} more")
        print()
