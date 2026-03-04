from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from action_labeler.labeler import ActionLabeler, LabelResult
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
        return [
            LabelResult(image_path=Path(), detection=d, response="stub")
            for d in detections
        ]


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
        _write_detections(det_file, ["0 0.5 0.5 0.3 0.4"])
        image = _make_image()

        labeler = _make_labeler()
        dets = labeler._load_detections(det_file, image)
        assert len(dets) == 1
        assert dets[0] == Detection(
            class_id=0, x_center=0.5, y_center=0.5, width=0.3, height=0.4,
            image_width=64, image_height=64,
        )

    def test_multiple_detections(self, tmp_path):
        det_file = tmp_path / "detect" / "img.txt"
        _write_detections(det_file, [
            "0 0.5 0.5 0.3 0.4",
            "1 0.2 0.8 0.1 0.2",
        ])
        image = _make_image()

        labeler = _make_labeler()
        dets = labeler._load_detections(det_file, image)
        assert len(dets) == 2
        assert dets[1].class_id == 1

    def test_missing_file_returns_empty(self, tmp_path):
        labeler = _make_labeler()
        dets = labeler._load_detections(tmp_path / "nonexistent.txt", _make_image())
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
    def test_applies_in_order(self):
        p1 = MagicMock()
        p1.process.side_effect = lambda img, dets: img.resize((32, 32))
        p2 = MagicMock()
        p2.process.side_effect = lambda img, dets: img.resize((16, 16))

        labeler = _make_labeler(preprocessors=[p1, p2])
        result = labeler._apply_preprocessors(_make_image(size=(64, 64)), [])
        assert result.size == (16, 16)

    def test_no_preprocessors_returns_original(self):
        labeler = _make_labeler()
        image = _make_image()
        result = labeler._apply_preprocessors(image, [])
        assert result is image


class TestRun:
    def test_collects_results_from_label(self, tmp_path):
        _write_image(tmp_path / "images" / "a.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt", ["0 0.5 0.5 0.3 0.4"]
        )

        labeler = _make_labeler()
        results = labeler.run(tmp_path)
        assert len(results) == 1
        assert results[0].image_path == tmp_path / "images" / "a.jpg"

    def test_skips_filtered_images(self, tmp_path):
        _write_image(tmp_path / "images" / "a.jpg")
        _write_detections(
            tmp_path / "detect" / "a.txt", ["0 0.5 0.5 0.3 0.4"]
        )

        reject_filter = MagicMock()
        reject_filter.filter.return_value = False
        labeler = _make_labeler(filters=[reject_filter])
        results = labeler.run(tmp_path)
        assert len(results) == 0

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
                return [
                    LabelResult(
                        image_path=Path(), detection=d, response="ok"
                    )
                    for d in detections
                ]

        model = MagicMock()
        model.load_image.side_effect = lambda img: img
        labeler = ErrorOnFirstLabeler(
            model=model, prompt=MagicMock()
        )
        results = labeler.run(tmp_path)

        # Second image still processed
        assert len(results) == 1
        captured = capsys.readouterr()
        assert "bad image" in captured.out

    def test_sets_image_path_on_results(self, tmp_path):
        _write_image(tmp_path / "images" / "photo.jpg")
        _write_detections(
            tmp_path / "detect" / "photo.txt",
            ["0 0.1 0.2 0.3 0.4", "1 0.5 0.6 0.7 0.8"],
        )

        labeler = _make_labeler()
        results = labeler.run(tmp_path)
        assert len(results) == 2
        for r in results:
            assert r.image_path == tmp_path / "images" / "photo.jpg"
