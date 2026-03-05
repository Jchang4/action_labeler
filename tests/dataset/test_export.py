from pathlib import Path

import yaml
import pytest
from PIL import Image

from action_labeler.dataset import Dataset, DatasetColumns
from action_labeler.types import Detection, LabelResult


def _make_detection(**kwargs) -> Detection:
    defaults = dict(
        class_id=0,
        x_center=0.5,
        y_center=0.5,
        width=0.3,
        height=0.4,
        image_width=64,
        image_height=64,
    )
    defaults.update(kwargs)
    return Detection(**defaults)


def _result(action: str) -> LabelResult:
    return LabelResult(action=action, response=action)


def _create_source_image(path: Path) -> None:
    """Create a minimal 1x1 PNG at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1)).save(path)


def _make_source_dataset(tmp_path: Path, folder: str, images: dict[str, list[tuple[Detection, str]]]) -> Dataset:
    """Build a Dataset with real image files on disk.

    Args:
        tmp_path: Pytest tmp_path fixture.
        folder: Action folder name (e.g., "sitting").
        images: Mapping of image filename to list of (detection, action) pairs.

    Returns:
        A Dataset with rows for all images/detections.
    """
    ds = Dataset()
    for filename, det_actions in images.items():
        image_path = tmp_path / folder / "images" / filename
        _create_source_image(image_path)
        dets = [da[0] for da in det_actions]
        results = [_result(da[1]) for da in det_actions]
        ds.add_rows(image_path, dets, results)
    return ds


class TestExportYolov8:
    def test_creates_directory_structure(self, tmp_path):
        ds = _make_source_dataset(tmp_path, "sitting", {
            "img.jpg": [(_make_detection(), "sitting")],
        })
        output = tmp_path / "output"
        ds.export_yolov8(output, seed=42)

        assert (output / "data.yaml").exists()
        assert (output / "train" / "images").is_dir()
        assert (output / "train" / "labels").is_dir()
        assert (output / "valid" / "images").is_dir()
        assert (output / "valid" / "labels").is_dir()

    def test_copies_images_with_prefix(self, tmp_path):
        ds = _make_source_dataset(tmp_path, "sitting", {
            "img.jpg": [(_make_detection(), "sitting")],
        })
        output = tmp_path / "output"
        ds.export_yolov8(output, val_ratio=0.0, seed=42)

        assert (output / "train" / "images" / "sitting_img.jpg").exists()

    def test_label_format(self, tmp_path):
        det = _make_detection(x_center=0.25, y_center=0.75, width=0.1, height=0.2)
        ds = _make_source_dataset(tmp_path, "sitting", {
            "img.jpg": [(det, "sitting")],
        })
        output = tmp_path / "output"
        ds.export_yolov8(output, val_ratio=0.0, seed=42)

        label_file = output / "train" / "labels" / "sitting_img.txt"
        assert label_file.exists()
        line = label_file.read_text().strip()
        assert line == "0 0.25 0.75 0.1 0.2"

    def test_multiple_detections_per_image(self, tmp_path):
        det1 = _make_detection(x_center=0.2, y_center=0.3, width=0.1, height=0.1)
        det2 = _make_detection(x_center=0.8, y_center=0.7, width=0.2, height=0.3, class_id=1)
        ds = _make_source_dataset(tmp_path, "sitting", {
            "img.jpg": [(det1, "sitting"), (det2, "standing")],
        })
        output = tmp_path / "output"
        ds.export_yolov8(output, val_ratio=0.0, seed=42)

        label_file = output / "train" / "labels" / "sitting_img.txt"
        lines = label_file.read_text().strip().split("\n")
        assert len(lines) == 2
        # "sitting" = class 0, "standing" = class 1 (alphabetical)
        assert lines[0].startswith("0 ")
        assert lines[1].startswith("1 ")

    def test_class_ids_alphabetical(self, tmp_path):
        ds = _make_source_dataset(tmp_path, "actions", {
            "a.jpg": [(_make_detection(), "walking")],
            "b.jpg": [(_make_detection(), "cycling")],
            "c.jpg": [(_make_detection(), "sitting")],
        })
        output = tmp_path / "output"
        result = ds.export_yolov8(output, val_ratio=0.0, seed=42)
        assert result["classes"] == {"cycling": 0, "sitting": 1, "walking": 2}

    def test_data_yaml_contents(self, tmp_path):
        ds = _make_source_dataset(tmp_path, "sitting", {
            "a.jpg": [(_make_detection(), "sitting")],
            "b.jpg": [(_make_detection(), "standing")],
        })
        output = tmp_path / "output"
        ds.export_yolov8(output, val_ratio=0.0, seed=42)

        with open(output / "data.yaml") as f:
            data = yaml.safe_load(f)

        assert data["names"] == ["sitting", "standing"]
        assert data["nc"] == 2
        assert data["path"] == "output"
        assert data["train"] == "train/images"
        assert data["val"] == "valid/images"

    def test_train_val_split_ratio(self, tmp_path):
        images = {f"img_{i}.jpg": [(_make_detection(), "sitting")] for i in range(20)}
        ds = _make_source_dataset(tmp_path, "sitting", images)
        output = tmp_path / "output"
        result = ds.export_yolov8(output, val_ratio=0.2, seed=42)

        assert result["train_images"] == 16
        assert result["val_images"] == 4

    def test_all_detections_for_image_in_same_split(self, tmp_path):
        det1 = _make_detection()
        det2 = _make_detection(class_id=1)
        ds = _make_source_dataset(tmp_path, "sitting", {
            "img.jpg": [(det1, "sitting"), (det2, "standing")],
        })
        output = tmp_path / "output"
        ds.export_yolov8(output, seed=42)

        # The image should appear in exactly one split
        in_train = (output / "train" / "images" / "sitting_img.jpg").exists()
        in_valid = (output / "valid" / "images" / "sitting_img.jpg").exists()
        assert in_train != in_valid  # exactly one

    def test_unique_names_from_different_folders(self, tmp_path):
        ds1 = _make_source_dataset(tmp_path, "sitting", {
            "img.jpg": [(_make_detection(), "sitting")],
        })
        ds2 = _make_source_dataset(tmp_path, "standing", {
            "img.jpg": [(_make_detection(), "standing")],
        })
        combined = Dataset.combine(ds1, ds2)
        output = tmp_path / "output"
        combined.export_yolov8(output, val_ratio=0.0, seed=42)

        assert (output / "train" / "images" / "sitting_img.jpg").exists()
        assert (output / "train" / "images" / "standing_img.jpg").exists()

    def test_missing_source_image_raises(self, tmp_path):
        ds = Dataset()
        fake_path = tmp_path / "sitting" / "images" / "missing.jpg"
        ds.add_rows(fake_path, [_make_detection()], [_result("sitting")])
        output = tmp_path / "output"
        with pytest.raises(FileNotFoundError, match="missing.jpg"):
            ds.export_yolov8(output, seed=42)

    def test_empty_dataset_raises(self, tmp_path):
        ds = Dataset()
        with pytest.raises(ValueError, match="Cannot export empty dataset"):
            ds.export_yolov8(tmp_path / "output")

    def test_overwrite_false_raises_if_exists(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        ds = _make_source_dataset(tmp_path, "sitting", {
            "img.jpg": [(_make_detection(), "sitting")],
        })
        with pytest.raises(FileExistsError):
            ds.export_yolov8(output, overwrite=False)

    def test_overwrite_true_replaces_dir(self, tmp_path):
        ds = _make_source_dataset(tmp_path, "sitting", {
            "img.jpg": [(_make_detection(), "sitting")],
        })
        output = tmp_path / "output"
        ds.export_yolov8(output, val_ratio=0.0, seed=42)
        # Add a marker file that should be gone after overwrite
        (output / "marker.txt").write_text("old")
        ds.export_yolov8(output, val_ratio=0.0, seed=42, overwrite=True)
        assert not (output / "marker.txt").exists()

    def test_seed_reproducibility(self, tmp_path):
        images = {f"img_{i}.jpg": [(_make_detection(), "sitting")] for i in range(10)}
        ds = _make_source_dataset(tmp_path, "sitting", images)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        r1 = ds.export_yolov8(out1, seed=123)
        r2 = ds.export_yolov8(out2, seed=123)

        train1 = sorted(f.name for f in (out1 / "train" / "images").iterdir())
        train2 = sorted(f.name for f in (out2 / "train" / "images").iterdir())
        assert train1 == train2
        assert r1["train_images"] == r2["train_images"]

    def test_stratification_rare_class_in_both_splits(self, tmp_path):
        # 18 sitting images, 2 walking images — walking should appear in both splits
        images = {}
        for i in range(18):
            images[f"sit_{i}.jpg"] = [(_make_detection(), "sitting")]
        for i in range(2):
            images[f"walk_{i}.jpg"] = [(_make_detection(), "walking")]
        ds = _make_source_dataset(tmp_path, "actions", images)
        output = tmp_path / "output"
        ds.export_yolov8(output, val_ratio=0.2, seed=42)

        # Check that walking appears in valid
        valid_labels = list((output / "valid" / "labels").iterdir())
        valid_actions = set()
        for lf in valid_labels:
            for line in lf.read_text().strip().split("\n"):
                class_id = int(line.split()[0])
                valid_actions.add(class_id)

        # class 1 = walking (alphabetical: sitting=0, walking=1)
        assert 1 in valid_actions

    def test_image_name_collision_raises(self, tmp_path):
        """Two different images from same-named parent folders produce collision."""
        ds = Dataset()
        # Manually create two paths that would collide
        path1 = tmp_path / "a" / "sitting" / "images" / "img.jpg"
        path2 = tmp_path / "b" / "sitting" / "images" / "img.jpg"
        _create_source_image(path1)
        _create_source_image(path2)
        ds.add_rows(path1, [_make_detection()], [_result("sitting")])
        ds.add_rows(path2, [_make_detection(class_id=1)], [_result("standing")])

        # Both produce "sitting_img.jpg"
        with pytest.raises(ValueError, match="collision"):
            ds.export_yolov8(tmp_path / "output")
