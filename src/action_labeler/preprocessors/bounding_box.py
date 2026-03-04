from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from action_labeler.preprocessors.base import BasePreprocessor

if TYPE_CHECKING:
    from action_labeler.types import Detection

COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 255, 128),    # mint
    (255, 0, 128),    # rose
    (128, 255, 0),    # lime
    (0, 128, 255),    # sky blue
    (255, 128, 128),  # salmon
    (128, 255, 128),  # light green
    (128, 128, 255),  # periwinkle
    (255, 255, 128),  # pastel yellow
    (255, 128, 255),  # pink
    (128, 255, 255),  # light cyan
    (200, 100, 50),   # rust
    (50, 100, 200),   # steel blue
]


class BoundingBox(BasePreprocessor):
    """Draw bounding boxes and detection index labels on the image.

    Each detection gets a unique color (cycling through a palette of 20).
    The detection index is drawn as a text label above the top-left corner
    of each box.
    """

    def __init__(self, line_width: int = 2, font_size: int = 16) -> None:
        self.line_width = line_width
        self.font_size = font_size

    def process(self, image: Image.Image, detections: list[Detection]) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        font = self._load_font()

        for i, det in enumerate(detections):
            color = COLORS[i % len(COLORS)]
            draw.rectangle(det.xyxy, outline=color, width=self.line_width)
            self._draw_label(draw, det, str(i), color, font)

        return image

    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        det: Detection,
        text: str,
        color: tuple[int, int, int],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 2

        bg_x1 = det.x1
        bg_y1 = max(0, det.y1 - text_h - 2 * padding)
        bg_x2 = det.x1 + text_w + 2 * padding
        bg_y2 = det.y1

        draw.rectangle((bg_x1, bg_y1, bg_x2, bg_y2), fill=color)
        draw.text((bg_x1 + padding, bg_y1 + padding), text, fill=(255, 255, 255), font=font)

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", self.font_size)
        except OSError:
            return ImageFont.load_default()
