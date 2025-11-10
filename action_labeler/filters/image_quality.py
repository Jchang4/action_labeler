import cv2
import numpy as np
from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter


class BrightnessFilter(IFilter):
    """Filter detections by brightness in the detection region.

    Image quality significantly affects classification. This filter helps
    identify detections that are too dark or too bright (overexposed).

    Args:
        min_brightness: Minimum average brightness (0-255)
        max_brightness: Maximum average brightness (0-255)

    Examples:
        # Exclude very dark detections
        BrightnessFilter(min_brightness=50, max_brightness=255)

        # Only well-lit detections
        BrightnessFilter(min_brightness=80, max_brightness=200)

        # Only dark/underexposed detections (for studying difficult cases)
        BrightnessFilter(min_brightness=0, max_brightness=60)
    """

    min_brightness: float
    max_brightness: float

    def __init__(self, min_brightness: float = 30, max_brightness: float = 225):
        if not 0 <= min_brightness <= 255:
            raise ValueError("min_brightness must be between 0 and 255")
        if not 0 <= max_brightness <= 255:
            raise ValueError("max_brightness must be between 0 and 255")
        if min_brightness > max_brightness:
            raise ValueError("min_brightness must be <= max_brightness")

        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        xyxy = detections.xyxy[index]
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

        # Crop to detection region
        cropped = image.crop((x1, y1, x2, y2))

        # Convert to grayscale for brightness calculation
        gray = cropped.convert("L")

        # Calculate average brightness
        brightness = float(np.array(gray).mean())

        return self.min_brightness <= brightness <= self.max_brightness


class BlurDetectionFilter(IFilter):
    """Filter blurry detections using variance of Laplacian.

    Blurry images are harder to classify. This filter uses the variance of
    the Laplacian operator as a measure of image sharpness.

    Args:
        min_sharpness: Minimum sharpness threshold (higher = sharper required)
                      Typical values: 100-500 for sharp images

    Examples:
        # Only sharp detections
        BlurDetectionFilter(min_sharpness=200)

        # Very strict sharpness requirement
        BlurDetectionFilter(min_sharpness=500)

        # Lenient (accept slightly blurry)
        BlurDetectionFilter(min_sharpness=50)

    Note:
        This filter requires additional dependencies (opencv-python or scipy).
        Falls back to always returning True if dependencies are not available.
    """

    min_sharpness: float

    def __init__(self, min_sharpness: float = 100.0):
        if min_sharpness < 0:
            raise ValueError("min_sharpness must be non-negative")

        self.min_sharpness = min_sharpness

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        # OpenCV implementation (preferred for performance)
        xyxy = detections.xyxy[index]
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

        # Crop to detection region and convert to grayscale
        cropped = image.crop((x1, y1, x2, y2))
        gray = np.array(cropped.convert("L"))

        # Compute Laplacian variance (measure of sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # type: ignore[attr-defined]
        sharpness = float(laplacian.var())

        return sharpness >= self.min_sharpness
