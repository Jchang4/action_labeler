from pathlib import Path

import pytest

from action_labeler.dataset import Dataset, DatasetColumns
from action_labeler.types import LabelResult
from action_labeler.types import Detection


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


def _make_dataset(entries: list[tuple[str, str]]) -> Dataset:
    """Build a dataset from (image_name, action) pairs."""
    ds = Dataset()
    for img, action in entries:
        ds.add_rows(
            Path(img),
            [_make_detection()],
            [LabelResult(action=action, response=action)],
        )
    return ds


def _make_multi_detection_dataset(
    entries: list[tuple[str, list[tuple[str, float]]]],
) -> Dataset:
    """Build a dataset where each entry is (image_name, [(action, x_center), ...]).

    x_center is varied so detections are unique within an image.
    """
    ds = Dataset()
    for img, det_specs in entries:
        detections = [_make_detection(x_center=x) for _, x in det_specs]
        results = [LabelResult(action=a, response=a) for a, _ in det_specs]
        ds.add_rows(Path(img), detections, results)
    return ds


class TestBalance:
    def test_perfect_balance(self):
        """3 classes with 100/200/300 detections → all ~100."""
        entries = (
            [(f"a{i}.jpg", "sitting") for i in range(100)]
            + [(f"b{i}.jpg", "standing") for i in range(200)]
            + [(f"c{i}.jpg", "walking") for i in range(300)]
        )
        ds = _make_dataset(entries)
        ds.balance(seed=42)
        counts = ds.df[DatasetColumns.ACTION].value_counts().to_dict()
        assert counts["sitting"] == 100
        assert counts["standing"] >= 100
        assert counts["walking"] >= 100
        # Standing and walking should be reduced from original
        assert counts["standing"] <= 200
        assert counts["walking"] <= 300

    def test_upsample_multiplier(self):
        """min=50, multiplier=1.5 → targets 75, capped by actual."""
        entries = (
            [(f"a{i}.jpg", "rare") for i in range(50)]
            + [(f"b{i}.jpg", "common") for i in range(200)]
        )
        ds = _make_dataset(entries)
        ds.balance(upsample=1.5, seed=42)
        counts = ds.df[DatasetColumns.ACTION].value_counts().to_dict()
        assert counts["rare"] == 50  # all kept
        assert counts["common"] >= 75  # target is ceil(50*1.5)=75
        assert counts["common"] <= 200

    def test_per_class_multiplier(self):
        """Dict with different multipliers per class."""
        entries = (
            [(f"a{i}.jpg", "rare") for i in range(50)]
            + [(f"b{i}.jpg", "mid") for i in range(150)]
            + [(f"c{i}.jpg", "common") for i in range(300)]
        )
        ds = _make_dataset(entries)
        ds.balance(upsample={"rare": 1.0, "mid": 2.0, "common": 1.5}, seed=42)
        counts = ds.df[DatasetColumns.ACTION].value_counts().to_dict()
        assert counts["rare"] == 50
        assert counts["mid"] >= 100  # target = ceil(50*2.0) = 100
        assert counts["common"] >= 75  # target = ceil(50*1.5) = 75

    def test_drops_whole_images(self):
        """No partial image removal — all detections for an image stay or go."""
        ds = _make_multi_detection_dataset([
            ("a.jpg", [("sitting", 0.1), ("standing", 0.2)]),
            ("b.jpg", [("sitting", 0.3)]),
            ("c.jpg", [("standing", 0.4)]),
            ("d.jpg", [("standing", 0.5)]),
        ])
        ds.balance(seed=42)
        # Each kept image should have all its original detections
        for img in ds.df[DatasetColumns.IMAGE_PATH].unique():
            img_rows = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == img]
            if img == Path("a.jpg"):
                assert len(img_rows) == 2

    def test_preserves_all_detections_for_kept_images(self):
        """Kept images retain every detection, not just the target action."""
        ds = _make_multi_detection_dataset([
            ("a.jpg", [("rare", 0.1), ("common", 0.2), ("common", 0.3)]),
            ("b.jpg", [("common", 0.4)]),
            ("c.jpg", [("common", 0.5)]),
        ])
        ds.balance(seed=42)
        # a.jpg must be kept (only source of "rare"), all 3 detections preserved
        a_rows = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("a.jpg")]
        assert len(a_rows) == 3

    def test_common_class_may_exceed_target(self):
        """Images kept for a rare class inflate the common class count."""
        ds = _make_multi_detection_dataset([
            ("a.jpg", [("rare", 0.1), ("common", 0.2), ("common", 0.3)]),
            ("b.jpg", [("common", 0.4)]),
        ])
        ds.balance(seed=42)
        counts = ds.df[DatasetColumns.ACTION].value_counts().to_dict()
        # rare has 1 detection, so min_count=1, target for common=1
        # But a.jpg (kept for rare) already contributes 2 common detections
        assert counts["rare"] == 1
        assert counts["common"] >= 2  # exceeds target of 1

    def test_seed_reproducibility(self):
        """Same seed produces the same result."""
        entries = (
            [(f"a{i}.jpg", "x") for i in range(20)]
            + [(f"b{i}.jpg", "y") for i in range(100)]
        )
        ds1 = _make_dataset(entries)
        ds2 = _make_dataset(entries)
        ds1.balance(seed=123)
        ds2.balance(seed=123)
        assert ds1.df[DatasetColumns.IMAGE_PATH].tolist() == ds2.df[DatasetColumns.IMAGE_PATH].tolist()

    def test_empty_dataset_noop(self):
        """No error on empty dataset."""
        ds = Dataset()
        ds.balance()
        assert len(ds) == 0

    def test_single_class(self):
        """Works with one action — effectively a random subsample."""
        entries = [(f"a{i}.jpg", "only") for i in range(50)]
        ds = _make_dataset(entries)
        ds.balance(seed=42)
        # target = min_count * 1.0 = 50, so all kept
        assert len(ds) == 50

    def test_multiplier_below_one(self):
        """Targets below min_count for further downsampling."""
        entries = (
            [(f"a{i}.jpg", "x") for i in range(100)]
            + [(f"b{i}.jpg", "y") for i in range(100)]
        )
        ds = _make_dataset(entries)
        ds.balance(upsample=0.5, seed=42)
        counts = ds.df[DatasetColumns.ACTION].value_counts().to_dict()
        # target = ceil(100 * 0.5) = 50
        assert counts["x"] >= 50
        assert counts["y"] >= 50
        assert counts["x"] <= 100
        assert counts["y"] <= 100


class TestRemoveClass:
    def test_drops_whole_image_by_default(self):
        """Default: drops the entire image if it has the target class."""
        ds = _make_multi_detection_dataset([
            ("a.jpg", [("walking", 0.1), ("sitting", 0.2)]),
            ("b.jpg", [("sitting", 0.3)]),
        ])
        ds.remove_class("walking")
        assert len(ds) == 1
        assert ds.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("b.jpg")

    def test_drops_all_images_with_class(self):
        ds = _make_dataset([
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
            ("c.jpg", "walking"),
        ])
        ds.remove_class("walking")
        assert len(ds) == 1
        assert ds.df[DatasetColumns.ACTION].iloc[0] == "sitting"

    def test_keeps_non_matching(self):
        ds = _make_dataset([
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
        ])
        ds.remove_class("running")
        assert len(ds) == 2

    def test_keep_image_only_drops_detections(self):
        """keep_image=True: only removes matching detections, keeps others."""
        ds = _make_multi_detection_dataset([
            ("a.jpg", [("walking", 0.1), ("sitting", 0.2)]),
            ("b.jpg", [("sitting", 0.3)]),
        ])
        ds.remove_class("walking", keep_image=True)
        assert len(ds) == 2
        actions = set(ds.df[DatasetColumns.ACTION])
        assert actions == {"sitting"}
        images = set(ds.df[DatasetColumns.IMAGE_PATH])
        assert images == {Path("a.jpg"), Path("b.jpg")}

    def test_keep_image_removes_image_if_only_class(self):
        """keep_image=True: image with only the target class is fully removed."""
        ds = _make_dataset([
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
        ])
        ds.remove_class("walking", keep_image=True)
        assert len(ds) == 1
        assert ds.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("b.jpg")


class TestRenameClass:
    def test_renames_matching_rows(self):
        ds = _make_dataset([
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
            ("c.jpg", "walking"),
        ])
        ds.rename_class("walking", "strolling")
        actions = ds.df[DatasetColumns.ACTION].tolist()
        assert actions == ["strolling", "sitting", "strolling"]

    def test_raises_if_same_name(self):
        ds = _make_dataset([("a.jpg", "walking")])
        with pytest.raises(ValueError, match="old_name and new_name are both"):
            ds.rename_class("walking", "walking")

    def test_raises_if_old_name_missing(self):
        ds = _make_dataset([("a.jpg", "walking")])
        with pytest.raises(ValueError, match="not found in dataset"):
            ds.rename_class("running", "jogging")

    def test_preserves_row_count(self):
        ds = _make_dataset([
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
        ])
        ds.rename_class("walking", "strolling")
        assert len(ds) == 2
