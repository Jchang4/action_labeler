from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from action_labeler.dataset import Dataset
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


def _make_dataset_with_images(tmp_path, entries: list[tuple[str, str]]) -> Dataset:
    """Build a dataset from (image_name, action) pairs, creating real image files."""
    ds = Dataset()
    for img_name, action in entries:
        img_path = tmp_path / img_name
        if not img_path.exists():
            Image.new("RGB", (64, 64), color="blue").save(img_path)
        ds.add_rows(img_path, [_make_detection()], [_result(action)])
    return ds


def _mock_subplots(rows, cols, **kwargs):
    """Return a mock fig and an array of mock axes."""
    fig = MagicMock()
    total = rows * cols
    axes_list = [MagicMock() for _ in range(total)]
    if total == 1:
        return fig, axes_list[0]
    axes = np.empty((rows, cols), dtype=object)
    for i, ax in enumerate(axes_list):
        axes[i // cols, i % cols] = ax
    return fig, axes


class TestPlotGrid:
    @patch("action_labeler.dataset.plot.plt")
    def test_shows_images(self, mock_plt, tmp_path):
        mock_plt.subplots.side_effect = _mock_subplots
        ds = _make_dataset_with_images(tmp_path, [
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
        ])
        ds.plot_grid(n=2, seed=42)
        mock_plt.show.assert_called_once()

    @patch("action_labeler.dataset.plot.plt")
    def test_filters_by_action(self, mock_plt, tmp_path):
        mock_plt.subplots.side_effect = _mock_subplots
        ds = _make_dataset_with_images(tmp_path, [
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
            ("c.jpg", "walking"),
        ])
        ds.plot_grid(n=10, action="walking", seed=42)
        mock_plt.show.assert_called_once()

    @patch("action_labeler.dataset.plot.plt")
    def test_empty_dataset_prints_message(self, mock_plt, capsys):
        ds = Dataset()
        ds.plot_grid()
        mock_plt.show.assert_not_called()
        assert "No rows" in capsys.readouterr().out


class TestPlotDistribution:
    @patch("action_labeler.dataset.plot.plt")
    def test_shows_bar_chart(self, mock_plt, tmp_path):
        ds = _make_dataset_with_images(tmp_path, [
            ("a.jpg", "walking"),
            ("b.jpg", "sitting"),
            ("c.jpg", "walking"),
        ])
        ds.plot_distribution()
        mock_plt.show.assert_called_once()

    @patch("action_labeler.dataset.plot.plt")
    def test_empty_dataset_prints_message(self, mock_plt, capsys):
        ds = Dataset()
        ds.plot_distribution()
        mock_plt.show.assert_not_called()
        assert "No rows" in capsys.readouterr().out


class TestDetectionStats:
    def test_returns_correct_columns(self, tmp_path):
        ds = _make_dataset_with_images(tmp_path, [("a.jpg", "walking")])
        result = ds.detection_stats()
        assert list(result.columns) == ["avg_width", "avg_height", "avg_area", "count"]

    def test_computes_averages(self):
        ds = Dataset()
        det1 = _make_detection(width=0.2, height=0.4)
        det2 = _make_detection(width=0.4, height=0.6, x_center=0.3)
        ds.add_rows(
            Path("a.jpg"),
            [det1, det2],
            [_result("walking"), _result("walking")],
        )
        result = ds.detection_stats()
        row = result.loc["walking"]
        assert abs(row["avg_width"] - 0.3) < 1e-6
        assert abs(row["avg_height"] - 0.5) < 1e-6
        assert abs(row["avg_area"] - (0.2 * 0.4 + 0.4 * 0.6) / 2) < 1e-6
        assert row["count"] == 2

    def test_sorted_by_area_descending(self):
        ds = Dataset()
        # small detections
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection(width=0.1, height=0.1)],
            [_result("small")],
        )
        # large detections
        ds.add_rows(
            Path("b.jpg"),
            [_make_detection(width=0.8, height=0.8)],
            [_result("large")],
        )
        result = ds.detection_stats()
        assert result.index[0] == "large"
        assert result.index[1] == "small"

    def test_multiple_actions(self):
        ds = Dataset()
        ds.add_rows(
            Path("a.jpg"),
            [_make_detection(width=0.3, height=0.4)],
            [_result("walking")],
        )
        ds.add_rows(
            Path("b.jpg"),
            [_make_detection(width=0.5, height=0.6)],
            [_result("sitting")],
        )
        result = ds.detection_stats()
        assert len(result) == 2
        assert set(result.index) == {"walking", "sitting"}
