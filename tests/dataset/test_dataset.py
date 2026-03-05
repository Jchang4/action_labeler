from pathlib import Path

import pandas as pd
import pytest
from action_labeler.dataset import Dataset, DatasetColumns
from action_labeler.types import LabelResult
from action_labeler.types import ActionResponse, Detection


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


class StubResponse(ActionResponse):
    confidence: float


def _result(action: str, response=None) -> LabelResult:
    """Shorthand to build a LabelResult."""
    if response is None:
        response = action
    return LabelResult(action=action, response=response)


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
        ds.add_rows(Path("a.jpg"), [det], [LabelResult(action="walking", response=resp)])
        assert len(ds) == 1
        assert ds.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("a.jpg")
        assert ds.df[DatasetColumns.DETECTION_INDEX].iloc[0] == 0
        assert ds.df[DatasetColumns.ACTION].iloc[0] == "walking"

    def test_detection_index_increments(self):
        ds = Dataset()
        det1 = _make_detection()
        det2 = _make_detection(class_id=1)
        ds.add_rows(
            Path("a.jpg"),
            [det1, det2],
            [_result("walking"), _result("sitting")],
        )
        assert list(ds.df[DatasetColumns.DETECTION_INDEX]) == [0, 1]

    def test_multiple_images(self):
        ds = Dataset()
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection(), _make_detection(class_id=1)],
            [_result("walking"), _result("sitting")],
        )
        ds.add_rows(
            Path("b.jpg"),
            [_make_detection()],
            [_result("running")],
        )
        assert len(ds) == 3
        a_rows = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("a.jpg")]
        assert list(a_rows[DatasetColumns.DETECTION_INDEX]) == [0, 1]
        b_rows = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("b.jpg")]
        assert list(b_rows[DatasetColumns.DETECTION_INDEX]) == [0]

    def test_deduplicates_keeping_latest(self):
        ds = Dataset()
        det = _make_detection()
        ds.add_rows(Path("a.jpg"), [det], [_result("old")])
        ds.add_rows(Path("a.jpg"), [det], [_result("new")])
        assert len(ds) == 1
        assert ds.df[DatasetColumns.ACTION].iloc[0] == "new"
        assert ds.df[DatasetColumns.DETECTION_INDEX].iloc[0] == 0

    def test_dedup_preserves_other_detections(self):
        ds = Dataset()
        det1 = _make_detection(class_id=0)
        det2 = _make_detection(class_id=1)
        ds.add_rows(Path("a.jpg"), [det1, det2], [_result("resp1"), _result("resp2")])
        # Overwrite only det1
        ds.add_rows(Path("a.jpg"), [det1], [_result("updated")])
        assert len(ds) == 2
        row0 = ds.df[ds.df[DatasetColumns.DETECTION].apply(lambda d: d.class_id == 0)]
        row1 = ds.df[ds.df[DatasetColumns.DETECTION].apply(lambda d: d.class_id == 1)]
        assert row0[DatasetColumns.ACTION].iloc[0] == "updated"
        assert row1[DatasetColumns.ACTION].iloc[0] == "resp2"
        assert list(ds.df[DatasetColumns.DETECTION_INDEX]) == [0, 1]

    def test_dedup_does_not_affect_other_images(self):
        ds = Dataset()
        det = _make_detection()
        ds.add_rows(Path("a.jpg"), [det], [_result("a_resp")])
        ds.add_rows(Path("b.jpg"), [det], [_result("b_resp")])
        assert len(ds) == 2
        ds.add_rows(Path("a.jpg"), [det], [_result("a_new")])
        assert len(ds) == 2
        a_row = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("a.jpg")]
        b_row = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("b.jpg")]
        assert a_row[DatasetColumns.ACTION].iloc[0] == "a_new"
        assert b_row[DatasetColumns.ACTION].iloc[0] == "b_resp"

    def test_action_column_populated(self):
        ds = Dataset()
        resp = StubResponse(action="walking", confidence=0.9)
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection()],
            [LabelResult(action="walking", response=resp)],
        )
        assert ds.df[DatasetColumns.ACTION].iloc[0] == "walking"
        assert ds.df[DatasetColumns.RESPONSE].iloc[0] == resp


class TestHasRow:
    def test_match_returns_true(self):
        ds = Dataset()
        det = _make_detection()
        ds.add_rows(Path("a.jpg"), [det], [_result("stub")])
        assert ds.has_row(Path("a.jpg"), det) is True

    def test_mismatch_returns_false(self):
        ds = Dataset()
        det = _make_detection()
        ds.add_rows(Path("a.jpg"), [det], [_result("stub")])
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
                DatasetColumns.ACTION: ["walking"],
                DatasetColumns.RESPONSE: ["stub"],
            }
        )
        ds = Dataset(df)
        assert len(ds) == 1


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        ds = Dataset()
        resp = StubResponse(action="walking", confidence=0.9)
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection()],
            [LabelResult(action="walking", response=resp)],
        )
        path = tmp_path / "dataset.pkl"
        ds.save(path)

        loaded = Dataset.load(path)
        assert len(loaded) == 1
        assert loaded.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("a.jpg")
        assert loaded.df[DatasetColumns.ACTION].iloc[0] == "walking"


class TestResponseField:
    def test_extracts_field(self):
        ds = Dataset()
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection()],
            [LabelResult(action="walking", response=StubResponse(action="walking", confidence=0.9))],
        )
        ds.add_rows(
            Path("b.jpg"),
            [_make_detection()],
            [LabelResult(action="sitting", response=StubResponse(action="sitting", confidence=0.8))],
        )
        actions = ds.response_field("action")
        assert list(actions) == ["walking", "sitting"]


class TestCombine:
    def test_combines_two_datasets(self):
        ds1 = Dataset()
        ds1.add_rows(Path("a.jpg"), [_make_detection()], [_result("walking")])
        ds2 = Dataset()
        ds2.add_rows(Path("b.jpg"), [_make_detection()], [_result("sitting")])
        combined = Dataset.combine(ds1, ds2)
        assert len(combined) == 2

    def test_no_args_returns_empty(self):
        combined = Dataset.combine()
        assert len(combined) == 0

    def test_single_dataset(self):
        ds = Dataset()
        ds.add_rows(Path("a.jpg"), [_make_detection()], [_result("walking")])
        combined = Dataset.combine(ds)
        assert len(combined) == 1

    def test_preserves_all_rows_no_dedup(self):
        ds1 = Dataset()
        det = _make_detection()
        ds1.add_rows(Path("a.jpg"), [det], [_result("walking")])
        ds2 = Dataset()
        ds2.add_rows(Path("a.jpg"), [det], [_result("sitting")])
        combined = Dataset.combine(ds1, ds2)
        assert len(combined) == 2

    def test_does_not_mutate_originals(self):
        ds1 = Dataset()
        ds1.add_rows(Path("a.jpg"), [_make_detection()], [_result("walking")])
        ds2 = Dataset()
        ds2.add_rows(Path("b.jpg"), [_make_detection()], [_result("sitting")])
        Dataset.combine(ds1, ds2)
        assert len(ds1) == 1
        assert len(ds2) == 1


class TestLen:
    def test_returns_row_count(self):
        ds = Dataset()
        ds.add_rows(Path("a.jpg"), [_make_detection()], [_result("stub")])
        ds.add_rows(Path("b.jpg"), [_make_detection()], [_result("stub")])
        assert len(ds) == 2


class TestRepr:
    def test_format(self):
        ds = Dataset()
        ds.add_rows(Path("a.jpg"), [_make_detection()], [_result("stub")])
        assert repr(ds) == "Dataset(1 rows)"
