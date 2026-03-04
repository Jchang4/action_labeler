from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from action_labeler.dataset import Dataset, DatasetColumns
from action_labeler.labeler import LabelResult
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


class TestFromLabelResults:
    def test_builds_dataframe(self):
        results = [
            LabelResult(
                image_path=Path("a.jpg"),
                detection=_make_detection(),
                response=StubResponse(action="walking", confidence=0.9),
            ),
        ]
        ds = Dataset.from_label_results(results)
        assert len(ds) == 1
        assert ds.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("a.jpg")
        assert ds.df[DatasetColumns.DETECTION_INDEX].iloc[0] == 0

    def test_detection_index_per_image(self):
        results = [
            LabelResult(
                image_path=Path("a.jpg"),
                detection=_make_detection(),
                response=StubResponse(action="walking", confidence=0.9),
            ),
            LabelResult(
                image_path=Path("a.jpg"),
                detection=_make_detection(class_id=1),
                response=StubResponse(action="sitting", confidence=0.8),
            ),
            LabelResult(
                image_path=Path("b.jpg"),
                detection=_make_detection(),
                response=StubResponse(action="running", confidence=0.7),
            ),
        ]
        ds = Dataset.from_label_results(results)
        assert len(ds) == 3
        # a.jpg gets indices 0, 1
        a_rows = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("a.jpg")]
        assert list(a_rows[DatasetColumns.DETECTION_INDEX]) == [0, 1]
        # b.jpg gets index 0
        b_rows = ds.df[ds.df[DatasetColumns.IMAGE_PATH] == Path("b.jpg")]
        assert list(b_rows[DatasetColumns.DETECTION_INDEX]) == [0]


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
        results = [
            LabelResult(
                image_path=Path("a.jpg"),
                detection=_make_detection(),
                response=StubResponse(action="walking", confidence=0.9),
            ),
        ]
        ds = Dataset.from_label_results(results)
        path = tmp_path / "dataset.pkl"
        ds.save(path)

        loaded = Dataset.load(path)
        assert len(loaded) == 1
        assert loaded.df[DatasetColumns.IMAGE_PATH].iloc[0] == Path("a.jpg")


class TestResponseField:
    def test_extracts_field(self):
        results = [
            LabelResult(
                image_path=Path("a.jpg"),
                detection=_make_detection(),
                response=StubResponse(action="walking", confidence=0.9),
            ),
            LabelResult(
                image_path=Path("b.jpg"),
                detection=_make_detection(),
                response=StubResponse(action="sitting", confidence=0.8),
            ),
        ]
        ds = Dataset.from_label_results(results)
        actions = ds.response_field("action")
        assert list(actions) == ["walking", "sitting"]


class TestLen:
    def test_returns_row_count(self):
        results = [
            LabelResult(
                image_path=Path("a.jpg"),
                detection=_make_detection(),
                response="stub",
            ),
            LabelResult(
                image_path=Path("b.jpg"),
                detection=_make_detection(),
                response="stub",
            ),
        ]
        ds = Dataset.from_label_results(results)
        assert len(ds) == 2


class TestRepr:
    def test_format(self):
        results = [
            LabelResult(
                image_path=Path("a.jpg"),
                detection=_make_detection(),
                response="stub",
            ),
        ]
        ds = Dataset.from_label_results(results)
        assert repr(ds) == "Dataset(1 rows)"
