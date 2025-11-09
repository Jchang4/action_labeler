from pathlib import Path

import numpy as np
from PIL import Image

from .helpers import (
    DetectionFormat,
    detect_format,
    keypoints_to_numpy,
    segmentation_points_to_xywh,
    xywh_to_segmentation_points,
    xywhs_to_xyxys,
    xyxys_to_xywhs,
    yolov8_labels_to_rows,
)


class Detection:
    """Container for YOLO detections in an image.

    Supports three YOLO formats: bounding boxes, segmentation, and pose estimation.
    Stores detections in pixel coordinates (xyxy) but can convert to/from normalized
    YOLO format (xywh).

    Attributes:
        xyxy: Bounding boxes in pixel coords, shape (N, 4) as [x1, y1, x2, y2]
        segmentation_points: List of polygons, each as list of normalized coords
        keypoints: Keypoints array, shape (N, K, 2) as [x, y]
        class_id: Class IDs, shape (N,)
        image: PIL Image object

    Reference:
        https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    """

    xyxy: np.ndarray
    segmentation_points: list[list[float]]
    keypoints: np.ndarray
    class_id: np.ndarray
    image: Image.Image

    def __init__(
        self,
        xyxy: np.ndarray,
        segmentation_points: list[list[float]],
        keypoints: np.ndarray,
        class_id: np.ndarray,
        image: Image.Image,
    ):
        """Initialize Detection container.

        Args:
            xyxy: Bounding boxes in pixels, shape (N, 4)
            segmentation_points: List of N polygons (can be empty lists for bbox-only)
            keypoints: Keypoints array, shape (N, K, 2) or empty array
            class_id: Class IDs, shape (N,)
            image: PIL.Image.Image object
        """
        # Validate inputs
        num_detections = len(xyxy)
        assert (
            len(segmentation_points) == num_detections
        ), f"Mismatch: {num_detections} detections but {len(segmentation_points)} segmentation_points"
        assert (
            len(class_id) == num_detections
        ), f"Mismatch: {num_detections} detections but {len(class_id)} class_ids"

        if keypoints.size > 0:
            assert (
                keypoints.shape[0] == num_detections
            ), f"Mismatch: {num_detections} detections but {keypoints.shape[0]} keypoint sets"

        assert xyxy.shape == (
            num_detections,
            4,
        ), f"Expected shape ({num_detections}, 4), got {xyxy.shape}"
        assert class_id.shape == (
            num_detections,
        ), f"Expected shape ({num_detections},), got {class_id.shape}"
        assert image.size[0] > 0 and image.size[1] > 0, "Image size must be positive"

        self.xyxy = xyxy
        self.segmentation_points = segmentation_points
        self.keypoints = keypoints
        self.class_id = class_id
        self.image = image

    @classmethod
    def from_text_path(
        cls,
        text_path: Path | str,
        image: Image.Image,
        num_keypoints: int | None = None,
    ) -> "Detection":
        """Load detections from YOLO format text file.

        Automatically detects format type (bbox, segment, or pose) based on
        the number of values in each row.

        Args:
            text_path: Path to YOLO format .txt file
            image: PIL.Image.Image object
            num_keypoints: Number of keypoints per detection (required for pose format)

        Returns:
            Detection object with appropriate format

        Raises:
            ValueError: If format cannot be determined or is invalid

        Reference:
            https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
        """
        rows = yolov8_labels_to_rows(text_path)
        if not rows:
            return cls.empty(image)

        # Detect format based on the rows
        format_type = detect_format(rows, num_keypoints)

        # Route to appropriate loader based on detected format
        if format_type == DetectionFormat.BBOX:
            return cls.from_bbox_text_path(text_path, image)
        elif format_type == DetectionFormat.SEGMENT:
            return cls.from_segment_text_path(text_path, image)
        elif format_type == DetectionFormat.POSE:
            if num_keypoints is None:
                raise ValueError(
                    f"num_keypoints is required for pose format detection in {text_path}"
                )
            return cls.from_pose_text_path(text_path, image, num_keypoints)
        else:
            raise ValueError(f"Unknown detection format: {format_type}")

    @classmethod
    def from_bbox_text_path(
        cls, text_path: Path | str, image: Image.Image
    ) -> "Detection":
        """Load bounding box detections from YOLO format text file.

        Format: class x_center y_center width height (5 values per line)

        Args:
            text_path: Path to .txt file
            image: PIL.Image.Image object

        Returns:
            Detection with bbox format

        Reference:
            https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
        """
        rows = yolov8_labels_to_rows(text_path)
        class_ids = [int(row[0]) for row in rows]
        xywhs = [tuple(row[1:5]) for row in rows]
        xyxys = xywhs_to_xyxys(xywhs, image.size)

        # Generate segmentation points from bbox (4 corners)
        segmentation_points = [xywh_to_segmentation_points(xywh) for xywh in xywhs]

        return cls(
            xyxy=np.array(xyxys).reshape(-1, 4),
            segmentation_points=segmentation_points,
            keypoints=np.array([]),  # No keypoints for bbox
            class_id=np.array(class_ids),
            image=image,
        )

    @classmethod
    def from_segment_text_path(
        cls, text_path: Path | str, image: Image.Image
    ) -> "Detection":
        """Load segmentation detections from YOLO format text file.

        Format: class x1 y1 x2 y2 ... xn yn (variable values per line)

        Args:
            text_path: Path to .txt file
            image: PIL.Image.Image object

        Returns:
            Detection with segment format

        Reference:
            https://docs.ultralytics.com/datasets/segment/
        """
        rows = yolov8_labels_to_rows(text_path)
        class_ids = [int(row[0]) for row in rows]
        segmentation_points = [row[1:] for row in rows]

        # Convert segmentation polygons to bounding boxes
        xywhs = [
            segmentation_points_to_xywh(seg_points)
            for seg_points in segmentation_points
        ]
        xyxys = xywhs_to_xyxys(xywhs, image.size)

        return cls(
            xyxy=np.array(xyxys).reshape(-1, 4),
            segmentation_points=segmentation_points,
            keypoints=np.array([]),  # No keypoints for segmentation
            class_id=np.array(class_ids),
            image=image,
        )

    @classmethod
    def from_pose_text_path(
        cls,
        text_path: Path | str,
        image: Image.Image,
        num_keypoints: int,
    ) -> "Detection":
        """Load pose estimation detections from YOLO format text file (2D format).

        Format: class x_center y_center width height px1 py1 px2 py2 ... pxn pyn

        Where:
        - First 5 values are bounding box (same as bbox format)
        - Remaining values are keypoints in pairs: (x, y)

        Args:
            text_path: Path to .txt file
            image: PIL.Image.Image object
            num_keypoints: Number of keypoints per detection

        Returns:
            Detection with pose format

        Reference:
            https://docs.ultralytics.com/datasets/pose/
        """
        rows = yolov8_labels_to_rows(text_path)
        class_ids = [int(row[0]) for row in rows]
        xywhs = [tuple(row[1:5]) for row in rows]
        xyxys = xywhs_to_xyxys(xywhs, image.size)

        # Generate segmentation points from bbox (4 corners)
        segmentation_points = [xywh_to_segmentation_points(xywh) for xywh in xywhs]

        # Extract keypoints (values after first 5)
        all_keypoints = []
        for row in rows:
            keypoints_flat = row[5:]  # Skip class and bbox
            keypoints_array = keypoints_to_numpy(keypoints_flat, num_keypoints)
            all_keypoints.append(keypoints_array)

        keypoints = np.array(all_keypoints)  # Shape: (N, num_keypoints, 2)

        return cls(
            xyxy=np.array(xyxys).reshape(-1, 4),
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=np.array(class_ids),
            image=image,
        )

    @classmethod
    def empty(
        cls,
        image: Image.Image | None = None,
    ) -> "Detection":
        """Create an empty Detection with no detections.

        Args:
            image: PIL.Image.Image object (optional, defaults to creating a 0x0 placeholder)

        Returns:
            Empty Detection object
        """
        if image is None:
            # Create a minimal placeholder image for empty detection
            image = Image.new('RGB', (0, 0))

        return cls(
            xyxy=np.array([]).reshape(-1, 4),
            segmentation_points=[],
            keypoints=np.array([]),
            class_id=np.array([]),
            image=image,
        )

    def get_index(self, index: int) -> "Detection":
        """Get a single detection by index.

        Args:
            index: Index of detection to extract

        Returns:
            Detection containing only the specified index
        """
        # Extract keypoints for this index if they exist
        if self.keypoints.size > 0:
            keypoints = self.keypoints[index : index + 1]  # Keep dims: (1, K, 2)
        else:
            keypoints = np.array([])

        return self.__class__(
            xyxy=np.array([self.xyxy[index]]).reshape(1, 4),
            segmentation_points=[self.segmentation_points[index]],
            keypoints=keypoints,
            class_id=np.array([self.class_id[index]]),
            image=self.image,
        )

    def copy(self) -> "Detection":
        """Create a deep copy of this Detection.

        Returns:
            Copy of this Detection
        """
        return self.__class__(
            xyxy=self.xyxy.copy(),
            segmentation_points=self.segmentation_points.copy(),
            keypoints=(
                self.keypoints.copy() if self.keypoints.size > 0 else np.array([])
            ),
            class_id=self.class_id.copy(),
            image=self.image.copy(),
        )

    def is_empty(self) -> bool:
        """Check if this Detection has no detections.

        Returns:
            True if no detections, False otherwise
        """
        return len(self.xyxy) == 0

    @property
    def xywh(self) -> list[tuple[float, float, float, float]]:
        """Get normalized xywh coordinates for all detections.

        Returns:
            List of (x_center, y_center, width, height) in normalized coords [0-1]
        """
        return xyxys_to_xywhs(self.xyxy, self.image.size)

    @property
    def image_size(self) -> tuple[int, int]:
        """Get image size as (width, height).

        This property provides backward compatibility for code that accesses image_size directly.

        Returns:
            Tuple of (width, height) in pixels
        """
        return self.image.size

    def __str__(self):
        return (
            f"<Detection "
            f"num_detections={len(self.xyxy)} "
            f"image_size={self.image.size}>"
        )

    def __repr__(self):
        return self.__str__()
