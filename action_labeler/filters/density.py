import numpy as np
from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.base import IFilter


class DetectionDensityFilter(IFilter):
    """Filter based on how crowded/overlapping the detection area is.

    Measures the maximum Intersection over Union (IoU) with other detections.
    Crowded scenes with overlapping detections are harder to classify due to
    occlusion and visual clutter.

    Args:
        max_overlap_ratio: Maximum allowed IoU with any other detection (0.0 to 1.0)
        include_crowded: If True, only include crowded detections.
                        If False (default), exclude crowded detections.

    Examples:
        # Exclude heavily overlapping detections (> 50% overlap)
        DetectionDensityFilter(max_overlap_ratio=0.5, include_crowded=False)

        # Only isolated detections (< 10% overlap)
        DetectionDensityFilter(max_overlap_ratio=0.1, include_crowded=False)

        # Only crowded detections (for studying occlusion)
        DetectionDensityFilter(max_overlap_ratio=0.3, include_crowded=True)
    """

    max_overlap_ratio: float
    include_crowded: bool

    def __init__(
        self,
        max_overlap_ratio: float = 0.5,
        include_crowded: bool = False,
    ):
        if not 0.0 <= max_overlap_ratio <= 1.0:
            raise ValueError("max_overlap_ratio must be between 0.0 and 1.0")

        self.max_overlap_ratio = max_overlap_ratio
        self.include_crowded = include_crowded

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        # If only one detection, it's not crowded
        if len(detections.xyxy) == 1:
            if self.include_crowded:
                return False  # Not crowded, so exclude
            else:
                return True  # Not crowded, so include

        # Calculate IoU with all other detections
        target_box = detections.xyxy[index]
        max_iou = 0.0

        for i, other_box in enumerate(detections.xyxy):
            if i == index:
                continue  # Skip self

            iou = self._calculate_iou(target_box, other_box)
            max_iou = max(max_iou, iou)

        # Check if detection is crowded
        is_crowded = max_iou > self.max_overlap_ratio

        # Return based on include_crowded setting
        if self.include_crowded:
            return is_crowded  # Include only crowded detections
        else:
            return not is_crowded  # Exclude crowded detections

    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two boxes.

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU value between 0.0 and 1.0
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No intersection

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return float(intersection_area) / float(union_area)


class DetectionSizeRankFilter(IFilter):
    """Filter based on detection size rank within the image.

    Salient objects (largest detections) may be easier to classify than
    background objects (smallest detections). This filter allows focusing
    on detections of a specific size rank.

    Args:
        rank: Which size rank to include - "largest", "smallest", "median"
        top_n: For "largest" rank, include top N largest detections (default: 1)
        bottom_n: For "smallest" rank, include bottom N smallest detections (default: 1)

    Examples:
        # Only the largest detection per image
        DetectionSizeRankFilter(rank="largest", top_n=1)

        # Only the 3 largest detections
        DetectionSizeRankFilter(rank="largest", top_n=3)

        # Only the smallest detection (likely background)
        DetectionSizeRankFilter(rank="smallest", bottom_n=1)

        # Only median-sized detections
        DetectionSizeRankFilter(rank="median")
    """

    rank: str
    top_n: int
    bottom_n: int

    VALID_RANKS = {"largest", "smallest", "median"}

    def __init__(
        self,
        rank: str = "largest",
        top_n: int = 1,
        bottom_n: int = 1,
    ):
        if rank not in self.VALID_RANKS:
            raise ValueError(f"rank must be one of {self.VALID_RANKS}, got '{rank}'")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if bottom_n < 1:
            raise ValueError("bottom_n must be at least 1")

        self.rank = rank
        self.top_n = top_n
        self.bottom_n = bottom_n

    def is_valid(
        self,
        image: Image.Image,
        index: int,
        detections: Detection,
    ) -> bool:
        # Calculate area for all detections
        areas = []
        for i, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = xyxy
            area = (x2 - x1) * (y2 - y1)
            areas.append((i, area))

        # Sort by area
        areas_sorted = sorted(areas, key=lambda x: x[1], reverse=True)

        if self.rank == "largest":
            # Check if index is in top N
            top_indices = [idx for idx, _ in areas_sorted[: self.top_n]]
            return index in top_indices

        elif self.rank == "smallest":
            # Check if index is in bottom N
            bottom_indices = [idx for idx, _ in areas_sorted[-self.bottom_n :]]
            return index in bottom_indices

        elif self.rank == "median":
            # Check if index is the median detection
            median_idx = len(areas_sorted) // 2
            median_detection_idx = areas_sorted[median_idx][0]
            return index == median_detection_idx

        return False
