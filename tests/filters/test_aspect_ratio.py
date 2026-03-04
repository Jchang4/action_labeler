from PIL import Image

from action_labeler.filters.aspect_ratio import AspectRatioFilter
from action_labeler.types import Detection


def _make_image() -> Image.Image:
    return Image.new("RGB", (100, 100))


def _make_detection(width: float, height: float) -> Detection:
    return Detection(
        class_id=0,
        x_center=0.5,
        y_center=0.5,
        width=width,
        height=height,
        image_width=100,
        image_height=100,
    )


class TestAspectRatioFilter:
    def test_keeps_normal_ratio(self):
        f = AspectRatioFilter(min_ratio=0.5, max_ratio=2.0)
        det = _make_detection(width=0.3, height=0.3)  # ratio = 1.0
        assert f.filter(_make_image(), [det]) is True

    def test_rejects_too_tall(self):
        f = AspectRatioFilter(min_ratio=0.5, max_ratio=2.0)
        det = _make_detection(width=0.1, height=0.5)  # ratio = 0.2
        assert f.filter(_make_image(), [det]) is False

    def test_rejects_too_wide(self):
        f = AspectRatioFilter(min_ratio=0.5, max_ratio=2.0)
        det = _make_detection(width=0.9, height=0.1)  # ratio = 9.0
        assert f.filter(_make_image(), [det]) is False

    def test_one_bad_detection_rejects_all(self):
        f = AspectRatioFilter(min_ratio=0.5, max_ratio=2.0)
        good = _make_detection(width=0.3, height=0.3)
        bad = _make_detection(width=0.1, height=0.5)
        assert f.filter(_make_image(), [good, bad]) is False

    def test_empty_detections_kept(self):
        f = AspectRatioFilter(min_ratio=0.5, max_ratio=2.0)
        assert f.filter(_make_image(), []) is True

    def test_zero_height_returns_inf(self):
        f = AspectRatioFilter(max_ratio=10.0)
        det = _make_detection(width=0.3, height=0.0)
        assert f.filter(_make_image(), [det]) is False

    def test_boundary_values_inclusive(self):
        f = AspectRatioFilter(min_ratio=0.5, max_ratio=2.0)
        at_min = _make_detection(width=0.2, height=0.4)  # ratio = 0.5
        at_max = _make_detection(width=0.4, height=0.2)  # ratio = 2.0
        assert f.filter(_make_image(), [at_min]) is True
        assert f.filter(_make_image(), [at_max]) is True

    def test_defaults_accept_everything(self):
        f = AspectRatioFilter()
        det = _make_detection(width=0.9, height=0.01)
        assert f.filter(_make_image(), [det]) is True
