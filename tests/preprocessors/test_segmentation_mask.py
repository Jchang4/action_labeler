import pytest
from PIL import Image

from action_labeler.preprocessors.segmentation_mask import (
    COLORS,
    SegmentationMask,
    _to_pixel_coords,
    _to_rgba,
)
from action_labeler.types import Detection


def _make_image(size=(200, 200)) -> Image.Image:
    return Image.new("RGB", size, color=(0, 0, 0))


def _det(
    segments: list[float] | None = None,
    x_center: float = 0.5,
    y_center: float = 0.5,
) -> Detection:
    return Detection(
        class_id=0,
        x_center=x_center,
        y_center=y_center,
        width=0.3,
        height=0.3,
        image_width=200,
        image_height=200,
        segments=segments,
    )


# Triangle covering roughly center of image
TRIANGLE = [0.3, 0.3, 0.7, 0.3, 0.5, 0.7]
SQUARE = [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]


class TestToRgba:
    def test_converts_named_color(self):
        assert _to_rgba("red", 128) == (255, 0, 0, 128)

    def test_full_opacity(self):
        assert _to_rgba("blue", 255) == (0, 0, 255, 255)

    def test_zero_opacity(self):
        assert _to_rgba("green", 0) == (0, 128, 0, 0)


class TestToPixelCoords:
    def test_converts_normalized_to_pixels(self):
        result = _to_pixel_coords([0.5, 0.5, 1.0, 1.0], 200, 100)
        assert result == [(100.0, 50.0), (200.0, 100.0)]

    def test_origin(self):
        result = _to_pixel_coords([0.0, 0.0], 200, 200)
        assert result == [(0.0, 0.0)]


class TestSegmentationMask:
    def test_returns_rgba_image(self):
        img = _make_image()
        result = SegmentationMask().process(img, [_det(TRIANGLE)])
        assert result.mode == "RGBA"

    def test_preserves_size(self):
        img = _make_image((400, 300))
        det = _det(TRIANGLE)
        det.image_width = 400
        det.image_height = 300
        result = SegmentationMask().process(img, [det])
        assert result.size == (400, 300)

    def test_draws_on_image(self):
        img = _make_image()
        original_rgba = img.convert("RGBA").tobytes()
        result = SegmentationMask().process(img, [_det(TRIANGLE)])
        assert result.tobytes() != original_rgba

    def test_original_unchanged(self):
        img = _make_image()
        original_data = img.tobytes()
        SegmentationMask().process(img, [_det(TRIANGLE)])
        assert img.tobytes() == original_data

    def test_no_segments_returns_unchanged(self):
        img = _make_image()
        result = SegmentationMask().process(img, [_det(segments=None)])
        assert result is img

    def test_empty_detections_returns_unchanged(self):
        img = _make_image()
        result = SegmentationMask().process(img, [])
        assert result is img

    def test_fill_differs_from_outline(self):
        img = _make_image()
        det = _det(SQUARE)
        outline = SegmentationMask(fill=False).process(img, [det])
        filled = SegmentationMask(fill=True).process(img, [det])
        assert outline.tobytes() != filled.tobytes()

    def test_different_opacities_differ(self):
        img = _make_image()
        det = _det(SQUARE)
        low = SegmentationMask(opacity=0.1, fill=True).process(img, [det])
        high = SegmentationMask(opacity=0.9, fill=True).process(img, [det])
        assert low.tobytes() != high.tobytes()

    def test_invalid_opacity_raises(self):
        with pytest.raises(ValueError, match="opacity"):
            SegmentationMask(opacity=1.5)
        with pytest.raises(ValueError, match="opacity"):
            SegmentationMask(opacity=-0.1)

    def test_multiple_detections(self):
        img = _make_image()
        dets = [
            _det(TRIANGLE, x_center=0.25),
            _det(SQUARE, x_center=0.75),
        ]
        result = SegmentationMask().process(img, dets)
        assert result.size == img.size

    def test_colors_cycle_beyond_palette(self):
        img = _make_image()
        n = len(COLORS) + 3
        dets = [_det(TRIANGLE) for _ in range(n)]
        # Should not raise
        SegmentationMask().process(img, dets)

    def test_custom_colors(self):
        img = _make_image()
        det = _det(SQUARE)
        result = SegmentationMask(colors=["blue"], fill=True).process(img, [det])
        default = SegmentationMask(fill=True).process(img, [det])
        assert result.tobytes() != default.tobytes()

    def test_skips_detections_without_segments(self):
        img = _make_image()
        dets = [_det(segments=None), _det(TRIANGLE)]
        original_rgba = img.convert("RGBA").tobytes()
        result = SegmentationMask().process(img, dets)
        # Should still draw the one with segments
        assert result.tobytes() != original_rgba
