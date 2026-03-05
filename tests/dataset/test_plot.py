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
