from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageColor, ImageDraw

from action_labeler.preprocessors.base import BasePreprocessor

if TYPE_CHECKING:
    from action_labeler.types import Detection

COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "orange",
    "purple",
    "lime",
    "pink",
    "teal",
    "coral",
    "salmon",
    "gold",
    "violet",
    "turquoise",
    "sienna",
    "khaki",
    "orchid",
    "steelblue",
]


def _to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    """Convert a color name to an RGBA tuple."""
    r, g, b = ImageColor.getrgb(color)
    return (r, g, b, alpha)


def _to_pixel_coords(
    segments: list[float], width: int, height: int
) -> list[tuple[float, float]]:
    """Convert normalized segment coordinates to pixel-space polygon points."""
    return [(x * width, y * height) for x, y in zip(segments[0::2], segments[1::2])]


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    outline_width: int,
    fill: bool,
) -> None:
    """Draw a single polygon on the overlay."""
    if fill:
        draw.polygon(polygon, fill=color, outline=color, width=outline_width)
    else:
        draw.polygon(polygon, outline=color, width=outline_width)


class SegmentationMask(BasePreprocessor):
    """Draw segmentation mask polygons on the image.

    Each detection's ``segments`` field provides the normalized polygon
    coordinates. Detections without segments are skipped.

    Args:
        opacity: Mask opacity from 0.0 (transparent) to 1.0 (opaque).
        outline_width: Width of the polygon outline in pixels.
        fill: Whether to fill the polygon interior.
        colors: List of color names to cycle through per detection.
            Defaults to a built-in 20-color palette.
    """

    def __init__(
        self,
        opacity: float = 0.3,
        outline_width: int = 4,
        fill: bool = False,
        colors: list[str] | None = None,
    ) -> None:
        if not (0.0 <= opacity <= 1.0):
            raise ValueError("opacity must be between 0.0 and 1.0")
        self.opacity = opacity
        self.outline_width = outline_width
        self.fill = fill
        self.colors = colors or COLORS

    def process(self, image: Image.Image, detections: list[Detection]) -> Image.Image:
        segments = [d for d in detections if d.segments is not None]
        if not segments:
            return image

        image = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        alpha = int(255 * self.opacity)
        width, height = image.size

        for i, det in enumerate(segments):
            color = _to_rgba(self.colors[i % len(self.colors)], alpha)
            polygon = _to_pixel_coords(det.segments, width, height)
            _draw_polygon(draw, polygon, color, self.outline_width, self.fill)

        return Image.alpha_composite(image, overlay)
