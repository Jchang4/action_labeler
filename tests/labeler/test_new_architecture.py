"""Smoke tests for new ActionLabeler architecture.

These tests verify basic functionality of the redesigned framework.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.labeler.core.experiment import ExperimentConfig
from action_labeler.labeler.core.image_provider import ImageData
from action_labeler.labeler.core.processing_modes import (
    BatchDetectionMode,
    SingleDetectionMode,
    get_processing_mode,
)
from action_labeler.labeler.core.processing_pipeline import (
    DefaultResponseParser,
    ModelResponse,
    ProcessingPipeline,
    ProcessingUnit,
)
from action_labeler.labeler.storage.label_store import LabelStore
from action_labeler.labeler.storage.metadata import LabeledDetection, LabelMetadata
from action_labeler.labeler.storage.persistence import LabelPersistence


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_create_experiment_config(self):
        """Test creating basic experiment config."""
        config = ExperimentConfig(
            name="test_experiment",
            model_name="gpt-4o-mini",
            prompt_template="What is the action?",
            classes=["walking", "running"],
        )

        assert config.name == "test_experiment"
        assert config.model_name == "gpt-4o-mini"
        assert config.processing_mode == "single"
        assert len(config.classes) == 2

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ExperimentConfig(
                name="",
                model_name="gpt-4o-mini",
                prompt_template="test",
                classes=["a"],
            )

        with pytest.raises(ValueError, match="Invalid processing_mode"):
            ExperimentConfig(
                name="test",
                model_name="gpt-4o-mini",
                prompt_template="test",
                classes=["a"],
                processing_mode="invalid",
            )

    def test_config_hash(self):
        """Test config hashing for deduplication."""
        config1 = ExperimentConfig(
            name="test",
            model_name="gpt-4o-mini",
            prompt_template="What?",
            classes=["a", "b"],
        )

        config2 = ExperimentConfig(
            name="test",  # Same config
            model_name="gpt-4o-mini",
            prompt_template="What?",
            classes=["a", "b"],
        )

        config3 = ExperimentConfig(
            name="test",
            model_name="gpt-4o-mini",
            prompt_template="What?",
            classes=["a", "b", "c"],  # Different classes
        )

        # Same config should have same hash
        assert config1.get_hash() == config2.get_hash()

        # Different config should have different hash
        assert config1.get_hash() != config3.get_hash()

    def test_config_serialization(self):
        """Test config save/load."""
        config = ExperimentConfig(
            name="test",
            model_name="gpt-4o-mini",
            prompt_template="What?",
            classes=["a", "b"],
            processing_mode="batch",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)

            loaded = ExperimentConfig.load(path)

            assert loaded.name == config.name
            assert loaded.model_name == config.model_name
            assert loaded.processing_mode == config.processing_mode
            assert loaded.classes == config.classes


class TestProcessingModes:
    """Test processing modes."""

    def test_single_detection_mode(self):
        """Test single detection mode creates one unit per detection."""
        mode = SingleDetectionMode()

        # Create test image and detections
        image = Image.new("RGB", (100, 100))
        detections = Detection(
            xyxy=np.array([[10, 10, 50, 50], [60, 60, 90, 90]]),
            segmentation_points=[[], []],
            keypoints=np.array([]),
            class_id=np.array([0, 0]),
            image=image,
        )

        units = mode.create_processing_units(image, "/test/image.jpg", detections)

        assert len(units) == 2  # One per detection
        assert units[0].detection_index == 0
        assert units[1].detection_index == 1
        assert mode.get_name() == "single"
        assert mode.requires_preprocessing() is True

    def test_batch_detection_mode(self):
        """Test batch detection mode creates single unit for all detections."""
        mode = BatchDetectionMode()

        image = Image.new("RGB", (100, 100))
        detections = Detection(
            xyxy=np.array([[10, 10, 50, 50], [60, 60, 90, 90]]),
            segmentation_points=[[], []],
            keypoints=np.array([]),
            class_id=np.array([0, 0]),
            image=image,
        )

        units = mode.create_processing_units(image, "/test/image.jpg", detections)

        assert len(units) == 1  # Single unit for all
        assert units[0].detection_index is None
        assert mode.get_name() == "batch"
        assert mode.requires_preprocessing() is False

    def test_get_processing_mode_factory(self):
        """Test processing mode factory function."""
        single = get_processing_mode("single")
        assert isinstance(single, SingleDetectionMode)

        batch = get_processing_mode("batch")
        assert isinstance(batch, BatchDetectionMode)

        with pytest.raises(ValueError, match="Invalid processing mode"):
            get_processing_mode("invalid")


class TestLabelStore:
    """Test label storage."""

    def test_create_empty_store(self):
        """Test creating empty label store."""
        store = LabelStore()
        assert len(store) == 0

    def test_add_label(self):
        """Test adding labels to store."""
        store = LabelStore()

        metadata = LabelMetadata(
            experiment_id="test_exp",
            model_name="gpt-4o-mini",
            prompt_version="1.0",
        )

        detection = LabeledDetection(
            image_path="/test/image.jpg",
            xywh=[0.5, 0.5, 0.2, 0.3],
            segmentation_points=[],
            label="walking",
            metadata=metadata,
        )

        added = store.add(detection)
        assert added is True
        assert len(store) == 1

        # Try adding duplicate
        added = store.add(detection)
        assert added is False  # Should reject duplicate
        assert len(store) == 1  # Count unchanged

    def test_deduplication(self):
        """Test deduplication works correctly."""
        store = LabelStore()

        metadata = LabelMetadata(
            experiment_id="test",
            model_name="gpt-4o-mini",
            prompt_version="1.0",
        )

        det1 = LabeledDetection(
            image_path="/test/img.jpg",
            xywh=[0.1, 0.2, 0.3, 0.4],
            segmentation_points=[],
            label="walking",
            metadata=metadata,
        )

        det2 = LabeledDetection(
            image_path="/test/img.jpg",
            xywh=[0.1, 0.2, 0.3, 0.4],  # Same xywh
            segmentation_points=[],
            label="running",  # Different label
            metadata=metadata,
        )

        det3 = LabeledDetection(
            image_path="/test/img.jpg",
            xywh=[0.5, 0.6, 0.3, 0.4],  # Different xywh
            segmentation_points=[],
            label="walking",
            metadata=metadata,
        )

        store.add(det1)
        assert len(store) == 1

        # Same image + xywh should be rejected
        added = store.add(det2)
        assert added is False
        assert len(store) == 1

        # Different xywh should be added
        added = store.add(det3)
        assert added is True
        assert len(store) == 2

    def test_filter_by_label(self):
        """Test filtering by label."""
        store = LabelStore()

        metadata = LabelMetadata(
            experiment_id="test",
            model_name="gpt-4o-mini",
            prompt_version="1.0",
        )

        for i, label in enumerate(["walking", "running", "walking"]):
            det = LabeledDetection(
                image_path=f"/test/img{i}.jpg",
                xywh=[0.5, 0.5, 0.2, 0.3],
                segmentation_points=[],
                label=label,
                metadata=metadata,
            )
            store.add(det)

        walking = store.filter_by_label("walking")
        assert len(walking) == 2

        running = store.filter_by_label("running")
        assert len(running) == 1

    def test_get_statistics(self):
        """Test statistics generation."""
        store = LabelStore()

        metadata = LabelMetadata(
            experiment_id="test",
            model_name="gpt-4o-mini",
            prompt_version="1.0",
        )

        for i in range(5):
            det = LabeledDetection(
                image_path=f"/test/img{i % 2}.jpg",  # 2 unique images
                xywh=[0.5 + i * 0.01, 0.5, 0.2, 0.3],
                segmentation_points=[],
                label=["walking", "running"][i % 2],
                metadata=metadata,
            )
            store.add(det)

        stats = store.get_statistics()

        assert stats["total_labels"] == 5
        assert stats["unique_images"] == 2
        assert stats["unique_labels"] == 2
        assert "walking" in stats["label_distribution"]
        assert "running" in stats["label_distribution"]


class TestLabelPersistence:
    """Test label persistence."""

    def test_save_load_pickle(self):
        """Test pickle save/load."""
        store = LabelStore()

        metadata = LabelMetadata(
            experiment_id="test",
            model_name="gpt-4o-mini",
            prompt_version="1.0",
        )

        det = LabeledDetection(
            image_path="/test/img.jpg",
            xywh=[0.5, 0.5, 0.2, 0.3],
            segmentation_points=[],
            label="walking",
            metadata=metadata,
        )
        store.add(det)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"

            # Save
            LabelPersistence.save_pickle(store, path)
            assert path.exists()

            # Load
            loaded = LabelPersistence.load_pickle(path)
            assert len(loaded) == 1

            loaded_det = loaded.get_all()[0]
            assert loaded_det.label == "walking"
            assert loaded_det.image_path == "/test/img.jpg"

    def test_save_load_json(self):
        """Test JSON save/load."""
        store = LabelStore()

        metadata = LabelMetadata(
            experiment_id="test",
            model_name="gpt-4o-mini",
            prompt_version="1.0",
        )

        det = LabeledDetection(
            image_path="/test/img.jpg",
            xywh=[0.5, 0.5, 0.2, 0.3],
            segmentation_points=[],
            label="walking",
            metadata=metadata,
        )
        store.add(det)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            # Save
            LabelPersistence.save_json(store, path)
            assert path.exists()

            # Verify it's valid JSON
            import json

            with open(path) as f:
                data = json.load(f)
            assert "detections" in data

            # Load
            loaded = LabelPersistence.load_json(path)
            assert len(loaded) == 1

    def test_auto_format_detection(self):
        """Test automatic format detection."""
        store = LabelStore()

        metadata = LabelMetadata(
            experiment_id="test",
            model_name="gpt-4o-mini",
            prompt_version="1.0",
        )

        det = LabeledDetection(
            image_path="/test/img.jpg",
            xywh=[0.5, 0.5, 0.2, 0.3],
            segmentation_points=[],
            label="walking",
            metadata=metadata,
        )
        store.add(det)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with auto-detect
            pkl_path = Path(tmpdir) / "test.pkl"
            LabelPersistence.save(store, pkl_path, format="auto")

            json_path = Path(tmpdir) / "test.json"
            LabelPersistence.save(store, json_path, format="auto")

            # Load with auto-detect
            loaded_pkl = LabelPersistence.load(pkl_path, format="auto")
            assert len(loaded_pkl) == 1

            loaded_json = LabelPersistence.load(json_path, format="auto")
            assert len(loaded_json) == 1


class TestProcessingPipeline:
    """Test processing pipeline."""

    def test_response_parser(self):
        """Test default response parser."""
        parser = DefaultResponseParser()

        unit = ProcessingUnit(
            image=Image.new("RGB", (100, 100)),
            detection=None,
            detection_index=0,
        )

        response = parser.parse("walking", unit)

        assert isinstance(response, ModelResponse)
        assert response.label == "walking"
        assert response.raw_response == "walking"
        assert response.is_valid is True

    def test_model_response_validation(self):
        """Test ModelResponse validation."""
        # Valid confidence
        response = ModelResponse(
            label="walking", raw_response="walking", confidence=0.95
        )
        assert response.confidence == 0.95

        # Invalid confidence (out of range)
        with pytest.raises(ValueError, match="Confidence must be in"):
            ModelResponse(label="walking", raw_response="walking", confidence=1.5)


class TestValidators:
    """Test label validators."""

    def test_strict_validator(self):
        """Test strict class validator."""
        from action_labeler.labeler.core.processing_pipeline import (
            StrictClassValidator,
        )

        validator = StrictClassValidator()
        classes = ["walking", "running", "sitting"]

        # Valid label
        is_valid, error = validator.validate("walking", classes)
        assert is_valid is True
        assert error is None

        # Valid label (case insensitive)
        is_valid, error = validator.validate("WALKING", classes)
        assert is_valid is True

        # Invalid label
        is_valid, error = validator.validate("jumping", classes)
        assert is_valid is False
        assert error is not None

    def test_flexible_validator(self):
        """Test flexible class validator."""
        from action_labeler.labeler.core.processing_pipeline import (
            FlexibleClassValidator,
        )

        validator = FlexibleClassValidator()
        classes = ["walking", "running"]

        # Exact match
        is_valid, error = validator.validate("walking", classes)
        assert is_valid is True

        # Contains match
        is_valid, error = validator.validate("person is walking", classes)
        assert is_valid is True

        # No match
        is_valid, error = validator.validate("jumping", classes)
        assert is_valid is False
