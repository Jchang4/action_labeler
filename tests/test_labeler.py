from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from action_labeler.dataset import Dataset
from action_labeler.labeler import ActionLabeler
from action_labeler.types import LabelResult
from action_labeler.types import Detection


def _make_image(mode="RGB", size=(64, 64)):
    return Image.new(mode, size, color="red")


def _write_image(path: Path, image: Image.Image | None = None):
    """Save an image file to disk."""
    image = image or _make_image()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_detections(path: Path, lines: list[str]):
    """Write YOLO-format detection lines to a txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


class StubLabeler(ActionLabeler):
    """Concrete subclass for testing the base class logic."""

    def label(self, image, detections):
        return [LabelResult(action="stub", response="stub") for d in detections]


def _make_labeler(**kwargs):
    defaults = {
        "model": MagicMock(),
        "prompt": MagicMock(),
        "preprocessors": None,
        "filters": None,
    }
    defaults.update(kwargs)
    # Make model.load_image return the image unchanged
    defaults["model"].load_image.side_effect = lambda img: img
    return StubLabeler(**defaults)


class TestLoadImages:
    def test_finds_jpg_jpeg_png(self, tmp_path):
        images_dir = tmp_path / "images"
        _write_image(images_dir / "a.jpg")
        _write_image(images_dir / "b.jpeg")
        _write_image(images_dir / "c.png")

        labeler = _make_labeler()
        paths = labeler._load_images(tmp_path)
        names = [p.name for p in paths]
        assert "a.jpg" in names
        assert "b.jpeg" in names
        assert "c.png" in names

    def test_ignores_non_image_files(self, tmp_path):
        images_dir = tmp_path / "images"
        _write_image(images_dir / "a.jpg")
        (images_dir / "notes.txt").write_text("hello")

        labeler = _make_labeler()
        paths = labeler._load_images(tmp_path)
        assert len(paths) == 1

    def test_returns_sorted(self, tmp_path):
        images_dir = tmp_path / "images"
        _write_image(images_dir / "c.jpg")
        _write_image(images_dir / "a.jpg")
        _write_image(images_dir / "b.jpg")

        labeler = _make_labeler()
        paths = labeler._load_images(tmp_path)
        names = [p.name for p in paths]
        assert names == ["a.jpg", "b.jpg", "c.jpg"]


class TestLoadDetections:
    def test_parses_yolo_format(self, tmp_path):
        det_file = tmp_path / "detect" / "img.txt"
        seg_file = tmp_path / "segments" / "img.txt"
        _write_detections(det_file, ["0 0.5 0.5 0.3 0.4"])
        image = _make_image()

        labeler = _make_labeler()
        dets = labeler._load_detections(det_file, seg_file, image)
        assert len(dets) == 1
        assert dets[0] == Detection(
            class_id=0, x_center=0.5, y_center=0.5, width=0.3, height=0.4,
            image_width=64, image_height=64,
        )

    def test_multiple_detections(self, tmp_path):
        det_file = tmp_path / "detect" / "img.txt"
        seg_file = tmp_path / "segments" / "img.txt"
        _write_detections(det_file, [
            "0 0.5 0.5 0.3 0.4",
            "1 0.2 0.8 0.1 0.2",
        ])
        image = _make_image()

        labeler = _make_labeler()
        dets = labeler._load_detections(det_file, seg_file, image)
        assert len(dets) == 2
        assert dets[1].class_id == 1

    def test_missing_file_returns_empty(self, tmp_path):
        labeler = _make_labeler()
        seg_file = tmp_path / "segments" / "nonexistent.txt"
        dets = labeler._load_detections(tmp_path / "nonexistent.txt", seg_file, _make_image())
        assert dets == []


class TestApplyFilters:
    def test_passes_with_no_filters(self):
        labeler = _make_labeler()
        assert labeler._apply_filters(_make_image(), []) is True

    def test_passes_when_all_accept(self):
        f1 = MagicMock()
        f1.filter.return_value = True
        f2 = MagicMock()
        f2.filter.return_value = True

        labeler = _make_labeler(filters=[f1, f2])
        assert labeler._apply_filters(_make_image(), []) is True

    def test_rejects_when_any_rejects(self):
        f1 = MagicMock()
        f1.filter.return_value = True
        f2 = MagicMock()
        f2.filter.return_value = False

        labeler = _make_labeler(filters=[f1, f2])
        assert labeler._apply_filters(_make_image(), []) is False


class TestApplyPreprocessors:
    def test_single_chain_applies_in_order(self):
        p1 = MagicMock()
        p1.process.side_effect = lambda img, dets: img.resize((32, 32))
        p2 = MagicMock()
        p2.process.side_effect = lambda img, dets: img.resize((16, 16))

        labeler = _make_labeler(preprocessors=[[p1, p2]])
        result = labeler._apply_preprocessors(_make_image(size=(64, 64)), [])
        assert len(result) == 1
        assert result[0].size == (16, 16)

    def test_multiple_chains_produce_multiple_images(self):
        p1 = MagicMock()
        p1.process.side_effect = lambda img, dets: img.resize((32, 32))
        p2 = MagicMock()
        p2.process.side_effect = lambda img, dets: img.resize((16, 16))

        labeler = _make_labeler(preprocessors=[[p1], [p2]])
        result = labeler._apply_preprocessors(_make_image(size=(64, 64)), [])
        assert len(result) == 2
        assert result[0].size == (32, 32)
        assert result[1].size == (16, 16)

    def test_no_preprocessors_returns_original(self):
        labeler = _make_labeler()
        image = _make_image()
        result = labeler._apply_preprocessors(image, [])
        assert len(result) == 1
        assert result[0] is image

    def test_chains_get_independent_copies(self):
        """Each chain operates on its own copy, not the original."""
        p1 = MagicMock()
        p1.process.side_effect = lambda img, dets: img.resize((32, 32))
        p2 = MagicMock()
        p2.process.side_effect = lambda img, dets: img.resize((16, 16))

        original = _make_image(size=(64, 64))
        labeler = _make_labeler(preprocessors=[[p1], [p2]])
        result = labeler._apply_preprocessors(original, [])
        # Original unchanged
        assert original.size == (64, 64)
        # Each chain independent
        assert result[0].size == (32, 32)
        assert result[1].size == (16, 16)


class TestRun:
    def test_returns_dataset(self, tmp_path):
        _write_image(tmp_path / "images" / "a.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt", ["0 0.5 0.5 0.3 0.4"]
        )

        labeler = _make_labeler()
        dataset = labeler.run(tmp_path)
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 1
        assert dataset.df["image_path"].iloc[0] == tmp_path / "images" / "a.jpg"

    def test_skips_filtered_images(self, tmp_path):
        _write_image(tmp_path / "images" / "a.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt", ["0 0.5 0.5 0.3 0.4"]
        )

        reject_filter = MagicMock()
        reject_filter.filter.return_value = False
        labeler = _make_labeler(filters=[reject_filter])
        dataset = labeler.run(tmp_path)
        assert len(dataset) == 0

    def test_continues_on_error(self, tmp_path, capsys):
        _write_image(tmp_path / "images" / "a.jpg")
        _write_image(tmp_path / "images" / "b.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt", ["0 0.5 0.5 0.3 0.4"]
        )
        _write_detections(
            tmp_path / "detect" / "b.txt", ["0 0.5 0.5 0.3 0.4"]
        )

        call_count = 0

        class ErrorOnFirstLabeler(ActionLabeler):
            def label(self, image, detections):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ValueError("bad image")
                return [LabelResult(action="ok", response="ok") for d in detections]

        model = MagicMock()
        model.load_image.side_effect = lambda img: img
        labeler = ErrorOnFirstLabeler(
            model=model, prompt=MagicMock()
        )
        dataset = labeler.run(tmp_path)

        # Second image still processed
        assert len(dataset) == 1
        captured = capsys.readouterr()
        assert "bad image" in captured.out

    def test_skips_fully_labeled_image(self, tmp_path):
        """When all detections in an image are already labeled, label() is not called."""
        _write_image(tmp_path / "images" / "a.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt",
            ["0 0.1 0.2 0.3 0.4", "1 0.5 0.6 0.7 0.8"],
        )

        label_calls = []

        class TrackingLabeler(ActionLabeler):
            def label(self, image, detections):
                label_calls.append(detections)
                return [LabelResult(action="resp", response="resp") for d in detections]

        model = MagicMock()
        model.load_image.side_effect = lambda img: img
        labeler = TrackingLabeler(model=model, prompt=MagicMock())

        # First run labels both detections
        dataset = labeler.run(tmp_path)
        assert len(dataset) == 2
        original_df = dataset.df.copy()

        # Resume — label() should not be called again
        label_calls.clear()
        dataset = labeler.run(tmp_path)
        assert len(label_calls) == 0
        # Dataset unchanged — same rows, same values
        assert len(dataset) == 2
        assert list(dataset.df["image_path"]) == list(original_df["image_path"])
        assert list(dataset.df["detection"]) == list(original_df["detection"])
        assert list(dataset.df["response"]) == list(original_df["response"])
        assert list(dataset.df["detection_index"]) == list(
            original_df["detection_index"]
        )

    def test_relabels_partially_labeled_image(self, tmp_path):
        """When some detections are already labeled, label() gets all detections
        and new responses overwrite old ones."""
        image_path = tmp_path / "images" / "a.jpg"
        _write_image(image_path)
        _write_detections(
            tmp_path / "detect" / "a.txt",
            ["0 0.1 0.2 0.3 0.4", "1 0.5 0.6 0.7 0.8"],
        )

        label_calls = []

        class TrackingLabeler(ActionLabeler):
            def label(self, image, detections):
                label_calls.append(detections)
                return [LabelResult(action="new_resp", response="new_resp") for d in detections]

        model = MagicMock()
        model.load_image.side_effect = lambda img: img
        labeler = TrackingLabeler(model=model, prompt=MagicMock())

        # First run — label both
        dataset = labeler.run(tmp_path)
        assert len(dataset) == 2

        # Keep only first detection, simulating a partial run
        labeler.dataset.df = labeler.dataset.df.iloc[:1].reset_index(drop=True)
        label_calls.clear()

        # Resume — label() receives ALL detections, not just the missing one
        dataset = labeler.run(tmp_path)

        assert len(label_calls) == 1
        assert len(label_calls[0]) == 2

        # Final dataset has both rows, old response overwritten by new
        assert len(dataset) == 2
        assert list(dataset.df["image_path"]) == [image_path, image_path]
        assert dataset.df["detection"].iloc[0].class_id == 0
        assert dataset.df["detection"].iloc[1].class_id == 1
        assert list(dataset.df["response"]) == ["new_resp", "new_resp"]
        assert list(dataset.df["detection_index"]) == [0, 1]

    def test_save_every_saves_periodically(self, tmp_path):
        """Dataset is saved every N newly-labeled images."""
        for name in ("a", "b", "c"):
            _write_image(tmp_path / "images" / f"{name}.jpg")
            _write_detections(
                tmp_path / "detect" / f"{name}.txt", ["0 0.5 0.5 0.3 0.4"]
            )

        save_path = tmp_path / "checkpoint.pkl"
        labeler = _make_labeler(save_every=2, save_path=save_path)
        labeler.run(tmp_path)

        # Should have saved: once at 2 images, and once at the end
        saved = Dataset.load(save_path)
        assert len(saved) == 3

    def test_save_every_without_save_path_raises(self):
        """save_every requires save_path."""
        import pytest
        with pytest.raises(ValueError, match="save_path is required"):
            _make_labeler(save_every=5)

    def test_final_save_on_completion(self, tmp_path):
        """Dataset is saved at the end of run() when save_path is set."""
        _write_image(tmp_path / "images" / "a.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt", ["0 0.5 0.5 0.3 0.4"]
        )

        save_path = tmp_path / "checkpoint.pkl"
        labeler = _make_labeler(save_path=save_path)
        labeler.run(tmp_path)

        saved = Dataset.load(save_path)
        assert len(saved) == 1

    def test_loads_existing_dataset_on_init(self, tmp_path):
        """When save_path points to an existing file, dataset is loaded on init."""
        _write_image(tmp_path / "images" / "a.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt", ["0 0.5 0.5 0.3 0.4"]
        )

        save_path = tmp_path / "checkpoint.pkl"

        # First labeler runs and saves
        labeler1 = _make_labeler(save_path=save_path)
        labeler1.run(tmp_path)
        assert len(labeler1.dataset) == 1

        # Second labeler loads existing checkpoint on init
        labeler2 = _make_labeler(save_path=save_path)
        assert len(labeler2.dataset) == 1

    def test_resumes_from_loaded_dataset(self, tmp_path):
        """A new labeler with save_path skips already-labeled images."""
        _write_image(tmp_path / "images" / "a.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt", ["0 0.5 0.5 0.3 0.4"]
        )

        save_path = tmp_path / "checkpoint.pkl"

        # First labeler runs and saves
        labeler1 = _make_labeler(save_path=save_path)
        labeler1.run(tmp_path)

        # Second labeler loads checkpoint, then runs — should skip the image
        label_calls = []

        class TrackingLabeler(ActionLabeler):
            def label(self, image, detections):
                label_calls.append(detections)
                return [LabelResult(action="x", response="x") for d in detections]

        model = MagicMock()
        model.load_image.side_effect = lambda img: img
        labeler2 = TrackingLabeler(
            model=model, prompt=MagicMock(), save_path=save_path
        )
        dataset = labeler2.run(tmp_path)

        assert len(label_calls) == 0
        assert len(dataset) == 1

    def test_sets_image_path_on_results(self, tmp_path):
        _write_image(tmp_path / "images" / "photo.jpg")
        _write_detections(
            tmp_path / "detect" / "photo.txt",
            ["0 0.1 0.2 0.3 0.4", "1 0.5 0.6 0.7 0.8"],
        )

        labeler = _make_labeler()
        dataset = labeler.run(tmp_path)
        assert len(dataset) == 2
        for path in dataset.df["image_path"]:
            assert path == tmp_path / "images" / "photo.jpg"
