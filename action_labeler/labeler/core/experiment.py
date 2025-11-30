"""Experiment tracking and versioning for labeling workflows.

This module provides experiment management for reproducible research workflows.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from action_labeler.dataclasses import DetectionType


@dataclass
class ExperimentConfig:
    """Configuration for a labeling experiment.

    This class captures all parameters that affect labeling results,
    enabling reproducible experiments and easy comparison.

    Attributes:
        name: Human-readable experiment name
        version: Experiment version (auto-incremented or manual)
        description: Optional description of experiment goals
        model_name: Name/ID of the VLM being used
        prompt_template: The prompt template string
        prompt_version: Version of the prompt (for tracking changes)
        classes: List of target classes
        detection_type: DETECT or SEGMENT
        processing_mode: "single", "batch", or "hybrid"
        filter_config: Configuration dict for filters
        preprocessor_config: Configuration dict for preprocessors
        metadata: Additional custom metadata
        created_at: Timestamp when experiment was created
    """

    name: str
    model_name: str
    prompt_template: str
    classes: list[str]
    version: str = "1.0"
    description: str = ""
    prompt_version: str = "1.0"
    detection_type: DetectionType = DetectionType.DETECT
    processing_mode: str = "single"  # "single", "batch", "hybrid"
    filter_config: dict[str, Any] = field(default_factory=dict)
    preprocessor_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not self.prompt_template:
            raise ValueError("Prompt template cannot be empty")
        if not self.classes:
            raise ValueError("Classes list cannot be empty")
        if self.processing_mode not in ["single", "batch", "hybrid"]:
            raise ValueError(
                f"Invalid processing_mode: {self.processing_mode}. "
                "Must be 'single', 'batch', or 'hybrid'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enum to string
        data["detection_type"] = self.detection_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        # Convert detection_type string to enum
        if "detection_type" in data and isinstance(data["detection_type"], str):
            data["detection_type"] = DetectionType(data["detection_type"])
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: str | Path) -> None:
        """Save experiment configuration to JSON file.

        Args:
            path: Path to save configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load experiment configuration from JSON file.

        Args:
            path: Path to configuration file

        Returns:
            Loaded ExperimentConfig
        """
        path = Path(path)
        return cls.from_json(path.read_text())

    def get_hash(self) -> str:
        """Get deterministic hash of configuration for deduplication.

        This hash includes all parameters that affect labeling results.
        Changes to name, description, or metadata don't affect the hash.

        Returns:
            Hex string hash of configuration
        """
        # Create dict with only parameters that affect results
        hashable_data = {
            "model_name": self.model_name,
            "prompt_template": self.prompt_template,
            "prompt_version": self.prompt_version,
            "classes": sorted(self.classes),  # Sort for consistency
            "detection_type": self.detection_type.value,
            "processing_mode": self.processing_mode,
            "filter_config": json.dumps(self.filter_config, sort_keys=True),
            "preprocessor_config": json.dumps(self.preprocessor_config, sort_keys=True),
        }

        # Create deterministic JSON string
        json_str = json.dumps(hashable_data, sort_keys=True)

        # Return SHA256 hash
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Experiment(name='{self.name}', version={self.version}, "
            f"model={self.model_name}, mode={self.processing_mode})"
        )


@dataclass
class ExperimentRun:
    """Represents a single execution of an experiment.

    Tracks when and how an experiment was run, including results location.

    Attributes:
        experiment_config: The experiment configuration used
        run_id: Unique identifier for this run
        started_at: When the run started
        completed_at: When the run completed (None if still running)
        results_path: Path to results file
        num_images_processed: Number of images processed
        num_labels_generated: Number of labels generated
        error_message: Error message if run failed
        status: "running", "completed", "failed"
    """

    experiment_config: ExperimentConfig
    run_id: str
    results_path: str | Path
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    num_images_processed: int = 0
    num_labels_generated: int = 0
    error_message: str | None = None
    status: str = "running"  # "running", "completed", "failed"

    def mark_completed(self, num_images: int, num_labels: int) -> None:
        """Mark run as completed with statistics.

        Args:
            num_images: Number of images processed
            num_labels: Number of labels generated
        """
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        self.num_images_processed = num_images
        self.num_labels_generated = num_labels

    def mark_failed(self, error_message: str) -> None:
        """Mark run as failed with error message.

        Args:
            error_message: Description of the failure
        """
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["experiment_config"] = self.experiment_config.to_dict()
        data["results_path"] = str(self.results_path)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRun":
        """Create from dictionary."""
        data["experiment_config"] = ExperimentConfig.from_dict(
            data["experiment_config"]
        )
        return cls(**data)


class ExperimentRegistry:
    """Registry for managing multiple experiments.

    Provides centralized management of experiment configurations and runs,
    enabling easy comparison and tracking across experiments.
    """

    def __init__(self, registry_dir: str | Path):
        """Initialize registry.

        Args:
            registry_dir: Directory to store experiment configs and metadata
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.registry_dir / "configs").mkdir(exist_ok=True)
        (self.registry_dir / "runs").mkdir(exist_ok=True)

    def register_experiment(self, config: ExperimentConfig) -> str:
        """Register a new experiment configuration.

        Args:
            config: Experiment configuration to register

        Returns:
            Experiment ID (hash of config)
        """
        experiment_id = config.get_hash()
        config_path = self.registry_dir / "configs" / f"{experiment_id}.json"

        # Save config if not already exists
        if not config_path.exists():
            config.save(config_path)

        return experiment_id

    def get_experiment(self, experiment_id: str) -> ExperimentConfig:
        """Retrieve experiment configuration by ID.

        Args:
            experiment_id: ID of experiment to retrieve

        Returns:
            Experiment configuration

        Raises:
            FileNotFoundError: If experiment ID not found
        """
        config_path = self.registry_dir / "configs" / f"{experiment_id}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found in registry")
        return ExperimentConfig.load(config_path)

    def list_experiments(self) -> list[tuple[str, ExperimentConfig]]:
        """List all registered experiments.

        Returns:
            List of (experiment_id, config) tuples
        """
        experiments = []
        config_dir = self.registry_dir / "configs"

        for config_path in sorted(config_dir.glob("*.json")):
            experiment_id = config_path.stem
            config = ExperimentConfig.load(config_path)
            experiments.append((experiment_id, config))

        return experiments

    def register_run(self, run: ExperimentRun) -> None:
        """Register an experiment run.

        Args:
            run: Experiment run to register
        """
        run_path = self.registry_dir / "runs" / f"{run.run_id}.json"
        run_path.write_text(json.dumps(run.to_dict(), indent=2))

    def get_runs_for_experiment(self, experiment_id: str) -> list[ExperimentRun]:
        """Get all runs for a specific experiment.

        Args:
            experiment_id: ID of experiment

        Returns:
            List of experiment runs, sorted by start time
        """
        runs = []
        runs_dir = self.registry_dir / "runs"

        for run_path in runs_dir.glob("*.json"):
            run_data = json.loads(run_path.read_text())
            run = ExperimentRun.from_dict(run_data)

            if run.experiment_config.get_hash() == experiment_id:
                runs.append(run)

        # Sort by start time
        runs.sort(key=lambda r: r.started_at)
        return runs
