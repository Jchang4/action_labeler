from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter


class EdgeProximityFilter(IFilter):
    """Filter detections based on proximity to image edges.

    Detections touching or near edges are often truncated/occluded, making them
    harder to classify. This filter can either include or exclude such detections.

    Args:
        min_distance_pixels: Minimum distance from any edge (in pixels)
        include_edge_detections: If True, only include edge detections.
                                If False (default), exclude edge detections.

    Examples:
        # Exclude detections within 10 pixels of any edge
        EdgeProximityFilter(min_distance_pixels=10, include_edge_detections=False)

        # Only include edge detections (for studying truncated objects)
        EdgeProximityFilter(min_distance_pixels=10, include_edge_detections=True)
    """

    min_distance_pixels: int
    include_edge_detections: bool

    def __init__(
        self,
        min_distance_pixels: int = 5,
        include_edge_detections: bool = False,
    ):
        if min_distance_pixels < 0:
            raise ValueError("min_distance_pixels must be non-negative")

        self.min_distance_pixels = min_distance_pixels
        self.include_edge_detections = include_edge_detections

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        xyxy = detections.xyxy[index]
        x1, y1, x2, y2 = xyxy
        img_width, img_height = image.size

        # Calculate distances to each edge
        dist_left = x1
        dist_top = y1
        dist_right = img_width - x2
        dist_bottom = img_height - y2

        # Find minimum distance to any edge
        min_edge_distance = min(dist_left, dist_top, dist_right, dist_bottom)

        # Check if detection is near edge
        is_near_edge = min_edge_distance < self.min_distance_pixels

        # Return based on include_edge_detections setting
        if self.include_edge_detections:
            return is_near_edge  # Include only edge detections
        else:
            return not is_near_edge  # Exclude edge detections


class CenterDetectionFilter(IFilter):
    """Filter detections based on position relative to image center.

    Useful for studying positional bias in classification - centered objects
    are typically easier to classify than objects at edges.

    Args:
        region: Region to include - "center", "edges", "top", "bottom", "left", "right"
        margin: Fraction of image dimension defining the region (0.0 to 0.5)

    Examples:
        # Only center detections (middle 40% of image)
        CenterDetectionFilter(region="center", margin=0.3)

        # Only edge detections (outer 20% of image)
        CenterDetectionFilter(region="edges", margin=0.2)

        # Only top half detections
        CenterDetectionFilter(region="top", margin=0.5)
    """

    region: str
    margin: float

    VALID_REGIONS = {"center", "edges", "top", "bottom", "left", "right"}

    def __init__(self, region: str = "center", margin: float = 0.3):
        if region not in self.VALID_REGIONS:
            raise ValueError(
                f"region must be one of {self.VALID_REGIONS}, got '{region}'"
            )
        if not 0.0 <= margin <= 0.5:
            raise ValueError("margin must be between 0.0 and 0.5")

        self.region = region
        self.margin = margin

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        xyxy = detections.xyxy[index]
        x1, y1, x2, y2 = xyxy
        img_width, img_height = image.size

        # Calculate detection center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Normalize to [0, 1]
        norm_x = center_x / img_width
        norm_y = center_y / img_height

        if self.region == "center":
            # Center region is within margin of image center (0.5, 0.5)
            return (
                0.5 - self.margin <= norm_x <= 0.5 + self.margin
                and 0.5 - self.margin <= norm_y <= 0.5 + self.margin
            )
        elif self.region == "edges":
            # Edges region is outside the center region
            return not (
                0.5 - self.margin <= norm_x <= 0.5 + self.margin
                and 0.5 - self.margin <= norm_y <= 0.5 + self.margin
            )
        elif self.region == "top":
            return norm_y <= self.margin
        elif self.region == "bottom":
            return norm_y >= 1.0 - self.margin
        elif self.region == "left":
            return norm_x <= self.margin
        elif self.region == "right":
            return norm_x >= 1.0 - self.margin

        return False
