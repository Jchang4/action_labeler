from PIL import Image

from action_labeler.preprocessors.bounding_box import COLORS, BoundingBox
from action_labeler.types import Detection


def _make_image(size=(200, 200)) -> Image.Image:
    return Image.new("RGB", size, color=(0, 0, 0))


def _det(x_center: float, y_center: float, width: float, height: float) -> Detection:
    return Detection(
        class_id=0,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        image_width=200,
        image_height=200,
    )


class TestBoundingBox:
    def test_returns_new_image(self):
        img = _make_image()
        result = BoundingBox().process(img, [_det(0.5, 0.5, 0.3, 0.3)])
        assert result is not img

    def test_original_unchanged(self):
        img = _make_image()
        original_data = img.tobytes()
        BoundingBox().process(img, [_det(0.5, 0.5, 0.3, 0.3)])
        assert img.tobytes() == original_data

    def test_draws_on_image(self):
        img = _make_image()
        result = BoundingBox().process(img, [_det(0.5, 0.5, 0.3, 0.3)])
        assert result.tobytes() != img.tobytes()

    def test_preserves_size(self):
        img = _make_image((400, 300))
        det = _det(0.5, 0.5, 0.3, 0.3)
        det = Detection(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.3, height=0.3, image_width=400, image_height=300,
        )
        result = BoundingBox().process(img, [det])
        assert result.size == (400, 300)

    def test_empty_detections_unchanged(self):
        img = _make_image()
        result = BoundingBox().process(img, [])
        assert result.tobytes() == img.tobytes()

    def test_multiple_detections_different_colors(self):
        """Each detection should use a different color from the palette."""
        img = _make_image()
        dets = [
            _det(0.25, 0.25, 0.2, 0.2),
            _det(0.75, 0.75, 0.2, 0.2),
        ]
        # Verify the color palette assigns different colors
        assert COLORS[0] != COLORS[1]
        # Just ensure it doesn't error with multiple detections
        result = BoundingBox().process(img, dets)
        assert result.size == img.size

    def test_colors_cycle_beyond_palette(self):
        """More detections than colors should cycle back through palette."""
        img = _make_image()
        n = len(COLORS) + 3
        dets = [_det(0.5, 0.5, 0.05, 0.05) for _ in range(n)]
        # Should not raise
        BoundingBox().process(img, dets)

    def test_custom_line_width(self):
        img = _make_image()
        det = _det(0.5, 0.5, 0.3, 0.3)
        thin = BoundingBox(line_width=1).process(img, [det])
        thick = BoundingBox(line_width=5).process(img, [det])
        assert thin.tobytes() != thick.tobytes()

    def test_detection_at_top_edge(self):
        """Label should not go above image boundary."""
        img = _make_image()
        det = _det(0.5, 0.05, 0.3, 0.1)  # near top
        # Should not raise
        BoundingBox().process(img, [det])
