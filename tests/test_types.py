from pathlib import Path

from PIL import Image

from action_labeler.types import Detection


def _make_image(size=(100, 200)):
    return Image.new("RGB", size, color="red")


class TestFromYolo:
    def test_parses_line(self):
        image = _make_image(size=(640, 480))
        det = Detection.from_yolo("0 0.5 0.5 0.3 0.4", image)
        assert det.class_id == 0
        assert det.x_center == 0.5
        assert det.y_center == 0.5
        assert det.width == 0.3
        assert det.height == 0.4
        assert det.image_width == 640
        assert det.image_height == 480

    def test_extracts_image_dimensions(self):
        image = _make_image(size=(1920, 1080))
        det = Detection.from_yolo("2 0.1 0.2 0.3 0.4", image)
        assert det.image_width == 1920
        assert det.image_height == 1080


class TestLoadTxt:
    def test_loads_multiple(self, tmp_path):
        det_file = tmp_path / "dets.txt"
        det_file.write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.8 0.1 0.2")
        image = _make_image(size=(640, 480))

        dets = Detection.load_txt(det_file, image)
        assert len(dets) == 2
        assert dets[0].class_id == 0
        assert dets[1].class_id == 1
        assert all(d.image_width == 640 for d in dets)

    def test_missing_file_returns_empty(self, tmp_path):
        image = _make_image()
        dets = Detection.load_txt(tmp_path / "nonexistent.txt", image)
        assert dets == []


class TestPixelProperties:
    def test_basic_computation(self):
        # 100x200 image, detection centered at (0.5, 0.5) with w=0.4, h=0.6
        det = Detection(
            class_id=0,
            x_center=0.5, y_center=0.5,
            width=0.4, height=0.6,
            image_width=100, image_height=200,
        )
        assert det.x1 == 30   # (0.5 - 0.2) * 100
        assert det.y1 == 40   # (0.5 - 0.3) * 200
        assert det.x2 == 70   # (0.5 + 0.2) * 100
        assert det.y2 == 160  # (0.5 + 0.3) * 200

    def test_clamped_to_image_bounds(self):
        det = Detection(
            class_id=0,
            x_center=0.0, y_center=0.0,
            width=0.5, height=0.5,
            image_width=100, image_height=100,
        )
        assert det.x1 == 0  # clamped, not negative
        assert det.y1 == 0
        assert det.x2 == 25
        assert det.y2 == 25

    def test_clamped_upper_bound(self):
        det = Detection(
            class_id=0,
            x_center=1.0, y_center=1.0,
            width=0.5, height=0.5,
            image_width=100, image_height=100,
        )
        assert det.x2 == 100  # clamped to image_width
        assert det.y2 == 100


class TestXyxy:
    def test_returns_correct_tuple(self):
        det = Detection(
            class_id=0,
            x_center=0.5, y_center=0.5,
            width=0.4, height=0.6,
            image_width=100, image_height=200,
        )
        assert det.xyxy == (30, 40, 70, 160)
