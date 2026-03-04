from PIL import Image

from action_labeler.filters.detection_count import DetectionCountFilter
from action_labeler.types import Detection


def _make_image() -> Image.Image:
    return Image.new("RGB", (100, 100))


def _make_detection() -> Detection:
    return Detection(
        class_id=0,
        x_center=0.5,
        y_center=0.5,
        width=0.3,
        height=0.3,
        image_width=100,
        image_height=100,
    )


class TestDetectionCountFilter:
    def test_within_range_kept(self):
        f = DetectionCountFilter(min_count=1, max_count=3)
        dets = [_make_detection(), _make_detection()]
        assert f.filter(_make_image(), dets) is True

    def test_too_few_rejected(self):
        f = DetectionCountFilter(min_count=2)
        assert f.filter(_make_image(), [_make_detection()]) is False

    def test_too_many_rejected(self):
        f = DetectionCountFilter(max_count=2)
        dets = [_make_detection() for _ in range(3)]
        assert f.filter(_make_image(), dets) is False

    def test_exact_min_kept(self):
        f = DetectionCountFilter(min_count=2)
        dets = [_make_detection(), _make_detection()]
        assert f.filter(_make_image(), dets) is True

    def test_exact_max_kept(self):
        f = DetectionCountFilter(max_count=2)
        dets = [_make_detection(), _make_detection()]
        assert f.filter(_make_image(), dets) is True

    def test_no_max_allows_any(self):
        f = DetectionCountFilter(min_count=1)
        dets = [_make_detection() for _ in range(100)]
        assert f.filter(_make_image(), dets) is True

    def test_empty_with_zero_min(self):
        f = DetectionCountFilter(min_count=0, max_count=5)
        assert f.filter(_make_image(), []) is True

    def test_defaults_accept_everything(self):
        f = DetectionCountFilter()
        assert f.filter(_make_image(), []) is True
        dets = [_make_detection() for _ in range(10)]
        assert f.filter(_make_image(), dets) is True
