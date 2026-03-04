from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from action_labeler.dataset import Dataset, DatasetColumns
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


class StubResponse(BaseModel):
    action: str
    confidence: float


class TestEmptyConstructor:
    def test_creates_valid_empty_dataset(self):
        ds = Dataset()
        assert len(ds) == 0
        assert set(ds.df.columns) >= DatasetColumns.REQUIRED


class TestAddRows:
    def test_single_image(self):
        ds = Dataset()
        det = _make_detection()
        resp = StubResponse(action="walking", confidence=0.9)
        ds.add_rows(Path("a.jpg"), [det], [resp])
        assert len(ds) == 1
        assert ds.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("a.jpg")
        assert ds.df[DatasetColumns.DETECTION_INDEX].iloc[0] == 0

    def test_detection_index_increments(self):
        ds = Dataset()
        det1 = _make_detection()
        det2 = _make_detection(class_id=1)
        ds.add_rows(
            Path("a.jpg"),
            [det1, det2],
            [
                StubResponse(action="walking", confidence=0.9),
                StubResponse(action="sitting", confidence=0.8),
            ],
        )
        assert list(ds.df[DatasetColumns.DETECTION_INDEX]) == [0, 1]

    def test_multiple_images(self):
        ds = Dataset()
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection(), _make_detection(class_id=1)],
            [
                StubResponse(action="walking", confidence=0.9),
                StubResponse(action="sitting", confidence=0.8),
            ],
        )
        ds.add_rows(
            Path("b.jpg"),
            [_make_detection()],
            [StubResponse(action="running", confidence=0.7)],
        )
        assert len(ds) == 3
        a_rows = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("a.jpg")]
        assert list(a_rows[DatasetColumns.DETECTION_INDEX]) == [0, 1]
        b_rows = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("b.jpg")]
        assert list(b_rows[DatasetColumns.DETECTION_INDEX]) == [0]


    def test_deduplicates_keeping_latest(self):
        ds = Dataset()
        det = _make_detection()
        ds.add_rows(Path("a.jpg"), [det], ["old_resp"])
        ds.add_rows(Path("a.jpg"), [det], ["new_resp"])
        assert len(ds) == 1
        assert ds.df[DatasetColumns.RESPONSE].iloc[0] == "new_resp"
        assert ds.df[DatasetColumns.DETECTION_INDEX].iloc[0] == 0

    def test_dedup_preserves_other_detections(self):
        ds = Dataset()
        det1 = _make_detection(class_id=0)
        det2 = _make_detection(class_id=1)
        ds.add_rows(Path("a.jpg"), [det1, det2], ["resp1", "resp2"])
        # Overwrite only det1
        ds.add_rows(Path("a.jpg"), [det1], ["updated"])
        assert len(ds) == 2
        row0 = ds.df[ds.df[DatasetColumns.DETECTION].apply(lambda d: d.class_id == 0)]
        row1 = ds.df[ds.df[DatasetColumns.DETECTION].apply(lambda d: d.class_id == 1)]
        assert row0[DatasetColumns.RESPONSE].iloc[0] == "updated"
        assert row1[DatasetColumns.RESPONSE].iloc[0] == "resp2"
        assert list(ds.df[DatasetColumns.DETECTION_INDEX]) == [0, 1]

    def test_dedup_does_not_affect_other_images(self):
        ds = Dataset()
        det = _make_detection()
        ds.add_rows(Path("a.jpg"), [det], ["a_resp"])
        ds.add_rows(Path("b.jpg"), [det], ["b_resp"])
        # Same detection in different images — no dedup
        assert len(ds) == 2
        # Overwrite a.jpg's detection
        ds.add_rows(Path("a.jpg"), [det], ["a_new"])
        assert len(ds) == 2
        a_row = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("a.jpg")]
        b_row = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("b.jpg")]
        assert a_row[DatasetColumns.RESPONSE].iloc[0] == "a_new"
        assert b_row[DatasetColumns.RESPONSE].iloc[0] == "b_resp"


class TestHasRow:
    def test_match_returns_true(self):
        ds = Dataset()
        det = _make_detection()
        ds.add_rows(Path("a.jpg"), [det], ["stub"])
        assert ds.has_row(Path("a.jpg"), det) is True

    def test_mismatch_returns_false(self):
        ds = Dataset()
        det = _make_detection()
        ds.add_rows(Path("a.jpg"), [det], ["stub"])
        other = _make_detection(class_id=99)
        assert ds.has_row(Path("a.jpg"), other) is False
        assert ds.has_row(Path("b.jpg"), det) is False


class TestValidate:
    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"image_path": [Path("a.jpg")]})
        with pytest.raises(ValueError, match="Missing columns"):
            Dataset(df)

    def test_passes_on_valid_df(self):
        df = pd.DataFrame(
            {
                DatasetColumns.IMAGE_PATH: [Path("a.jpg")],
                DatasetColumns.DETECTION_INDEX: [0],
                DatasetColumns.DETECTION: [_make_detection()],
                DatasetColumns.RESPONSE: ["stub"],
            }
        )
        ds = Dataset(df)
        assert len(ds) == 1


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        ds = Dataset()
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection()],
            [StubResponse(action="walking", confidence=0.9)],
        )
        path = tmp_path / "dataset.pkl"
        ds.save(path)

        loaded = Dataset.load(path)
        assert len(loaded) == 1
        assert loaded.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("a.jpg")


class TestResponseField:
    def test_extracts_field(self):
        ds = Dataset()
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection()],
            [StubResponse(action="walking", confidence=0.9)],
        )
        ds.add_rows(
            Path("b.jpg"),
            [_make_detection()],
            [StubResponse(action="sitting", confidence=0.8)],
        )
        actions = ds.response_field("action")
        assert list(actions) == ["walking", "sitting"]


class TestLen:
    def test_returns_row_count(self):
        ds = Dataset()
        ds.add_rows(Path("a.jpg"), [_make_detection()], ["stub"])
        ds.add_rows(Path("b.jpg"), [_make_detection()], ["stub"])
        assert len(ds) == 2


class TestRepr:
    def test_format(self):
        ds = Dataset()
        ds.add_rows(Path("a.jpg"), [_make_detection()], ["stub"])
        assert repr(ds) == "Dataset(1 rows)"
