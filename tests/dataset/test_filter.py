from pathlib import Path

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


class TestRemoveClass:
    def test_removes_matching_rows(self):
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


class TestKeepClasses:
    def test_keeps_only_specified(self):
        ds = _make_dataset([
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
            ("c.jpg", "running"),
        ])
        ds.keep_classes(["walking", "running"])
        assert len(ds) == 2
        actions = set(ds.df[DatasetColumns.ACTION])
        assert actions == {"walking", "running"}

    def test_removes_unspecified(self):
        ds = _make_dataset([
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
        ])
        ds.keep_classes(["running"])
        assert len(ds) == 0


class TestRemoveImage:
    def test_removes_all_rows_for_image(self):
        ds = _make_dataset([
            ("a.jpg", "walking"),
            ("a.jpg", "sitting"),
            ("b.jpg", "running"),
        ])
        ds.remove_image(Path("a.jpg"))
        assert len(ds) == 1
        assert ds.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("b.jpg")
