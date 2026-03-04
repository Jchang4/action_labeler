from PIL import Image

from action_labeler.filters.overlap import OverlapFilter
from action_labeler.types import Detection


def _make_image() -> Image.Image:
    return Image.new("RGB", (100, 100))


def _det(x_center: float, y_center: float, width: float, height: float) -> Detection:
    return Detection(
        class_id=0,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        image_width=100,
        image_height=100,
    )


class TestOverlapFilter:
    def test_no_overlap_kept(self):
        f = OverlapFilter(max_iou=0.5)
        a = _det(0.2, 0.5, 0.2, 0.2)  # left side
        b = _det(0.8, 0.5, 0.2, 0.2)  # right side
        assert f.filter(_make_image(), [a, b]) is True

    def test_identical_boxes_rejected(self):
        f = OverlapFilter(max_iou=0.5)
        a = _det(0.5, 0.5, 0.3, 0.3)
        b = _det(0.5, 0.5, 0.3, 0.3)
        assert f.filter(_make_image(), [a, b]) is False

    def test_high_overlap_rejected(self):
        f = OverlapFilter(max_iou=0.3)
        a = _det(0.5, 0.5, 0.4, 0.4)
        b = _det(0.55, 0.55, 0.4, 0.4)  # mostly overlapping
        assert f.filter(_make_image(), [a, b]) is False

    def test_single_detection_kept(self):
        f = OverlapFilter(max_iou=0.0)
        a = _det(0.5, 0.5, 0.3, 0.3)
        assert f.filter(_make_image(), [a]) is True

    def test_empty_detections_kept(self):
        f = OverlapFilter(max_iou=0.0)
        assert f.filter(_make_image(), []) is True

    def test_three_boxes_one_bad_pair(self):
        f = OverlapFilter(max_iou=0.5)
        a = _det(0.2, 0.5, 0.2, 0.2)
        b = _det(0.8, 0.5, 0.2, 0.2)
        c = _det(0.8, 0.5, 0.2, 0.2)  # identical to b
        assert f.filter(_make_image(), [a, b, c]) is False

    def test_partial_overlap_below_threshold(self):
        f = OverlapFilter(max_iou=0.5)
        a = _det(0.3, 0.5, 0.2, 0.2)
        b = _det(0.45, 0.5, 0.2, 0.2)  # slight overlap
        assert f.filter(_make_image(), [a, b]) is True
