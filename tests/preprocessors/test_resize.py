from PIL import Image

from action_labeler.preprocessors.resize import Resize
from action_labeler.types import Detection


def _make_detection() -> Detection:
    return Detection(
        class_id=0, x_center=0.5, y_center=0.5,
        width=0.3, height=0.3, image_width=100, image_height=100,
    )


class TestResize:
    def test_landscape_image(self):
        img = Image.new("RGB", (800, 400))
        result = Resize(400).process(img, [_make_detection()])
        assert result.size == (400, 200)

    def test_portrait_image(self):
        img = Image.new("RGB", (400, 800))
        result = Resize(400).process(img, [_make_detection()])
        assert result.size == (200, 400)

    def test_square_image(self):
        img = Image.new("RGB", (600, 600))
        result = Resize(300).process(img, [_make_detection()])
        assert result.size == (300, 300)

    def test_already_at_size_unchanged(self):
        img = Image.new("RGB", (400, 300))
        result = Resize(400).process(img, [_make_detection()])
        assert result is img

    def test_smaller_than_size_unchanged(self):
        img = Image.new("RGB", (200, 100))
        result = Resize(400).process(img, [_make_detection()])
        assert result is img

    def test_preserves_aspect_ratio(self):
        img = Image.new("RGB", (1920, 1080))
        result = Resize(960).process(img, [_make_detection()])
        w, h = result.size
        original_ratio = 1920 / 1080
        new_ratio = w / h
        assert abs(original_ratio - new_ratio) < 0.01
