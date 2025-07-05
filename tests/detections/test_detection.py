import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from action_labeler.detections.detection import Detection
from action_labeler.helpers.detections_helpers import xywh_to_segmentation_points


@pytest.fixture
def detection_text_file():
    """Create a temporary detection text file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        # Format: class_id, x_center, y_center, width, height
        f.write(b"0 0.5 0.5 0.2 0.3\n")  # Center box
        f.write(b"1 0.2 0.2 0.1 0.1\n")  # Top-left box
        f.write(b"2 0.8 0.8 0.2 0.1\n")  # Bottom-right box
        temp_path = f.name

    yield Path(temp_path)

    # Clean up
    os.unlink(temp_path)


@pytest.fixture
def segmentation_text_file():
    """Create a temporary segmentation text file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        # Format: class_id, x1, y1, x2, y2, ..., xn, yn
        # Rectangle with center at (0.5, 0.5), width=0.2, height=0.3
        points = xywh_to_segmentation_points((0.5, 0.5, 0.2, 0.3))
        f.write(f"0 {' '.join(map(str, points))}\n".encode())

        # Rectangle with center at (0.2, 0.2), width=0.1, height=0.1
        points = xywh_to_segmentation_points((0.2, 0.2, 0.1, 0.1))
        f.write(f"1 {' '.join(map(str, points))}\n".encode())

        # Rectangle with center at (0.8, 0.8), width=0.2, height=0.1
        points = xywh_to_segmentation_points((0.8, 0.8, 0.2, 0.1))
        f.write(f"2 {' '.join(map(str, points))}\n".encode())

        temp_path = f.name

    yield Path(temp_path)

    # Clean up
    os.unlink(temp_path)


class TestDetection:
    """Test cases for the Detection class."""

    def test_empty_detection(self):
        """Test creating an empty detection."""
        detection = Detection.empty(image_size=(640, 480))

        assert detection.is_empty()
        assert detection.image_size == (640, 480)
        assert len(detection.xyxy) == 0
        assert len(detection.segmentation_points) == 0
        assert len(detection.class_id) == 0

    def test_from_detection_text_path(self, detection_text_file):
        """Test creating a Detection from a detection text file."""
        image_size = (640, 480)
        detection = Detection.from_detection_text_path(detection_text_file, image_size)

        assert not detection.is_empty()
        assert detection.image_size == image_size
        assert len(detection.xyxy) == 3
        assert len(detection.segmentation_points) == 3
        assert len(detection.class_id) == 3

        # Check class IDs
        np.testing.assert_array_equal(detection.class_id, [0, 1, 2])

        # Check first detection (center box)
        # Expected xyxy for (0.5, 0.5, 0.2, 0.3) in a 640x480 image
        expected_xyxy = np.array(
            [
                [256, 168, 384, 312],
                [96, 72, 160, 120],
                [448, 360, 576, 408],
            ]
        )

        # Allow for small floating point differences
        np.testing.assert_allclose(detection.xyxy, expected_xyxy, rtol=1e-5)

        # Check that segmentation_points is a list of lists
        assert isinstance(detection.segmentation_points, list)
        assert all(isinstance(points, list) for points in detection.segmentation_points)

        # Check the first segmentation points
        expected_points = xywh_to_segmentation_points((0.5, 0.5, 0.2, 0.3))
        assert np.allclose(detection.segmentation_points[0], expected_points)

    def test_from_segmentation_text_path(self, segmentation_text_file):
        """Test creating a Detection from a segmentation text file."""
        image_size = (640, 480)
        detection = Detection.from_segmentation_text_path(
            segmentation_text_file, image_size
        )

        assert not detection.is_empty()
        assert detection.image_size == image_size
        assert len(detection.xyxy) == 3
        assert len(detection.segmentation_points) == 3
        assert len(detection.class_id) == 3

        # Check class IDs
        np.testing.assert_array_equal(detection.class_id, [0, 1, 2])

        # Check that segmentation_points is a list of lists
        assert isinstance(detection.segmentation_points, list)
        assert all(isinstance(points, list) for points in detection.segmentation_points)

    def test_from_text_path_detection(self, detection_text_file):
        """Test creating a Detection using from_text_path with a detection file."""
        image_size = (640, 480)
        detection = Detection.from_text_path(detection_text_file, image_size)

        assert not detection.is_empty()
        assert len(detection.class_id) == 3

    def test_from_text_path_segmentation(self, segmentation_text_file):
        """Test creating a Detection using from_text_path with a segmentation file."""
        image_size = (640, 480)
        detection = Detection.from_text_path(segmentation_text_file, image_size)

        assert not detection.is_empty()
        assert len(detection.class_id) == 3

    def test_copy(self, detection_text_file):
        """Test copying a Detection object."""
        image_size = (640, 480)
        detection = Detection.from_detection_text_path(detection_text_file, image_size)
        copied = detection.copy()

        # Check that it's a different object but with the same values
        assert detection is not copied
        np.testing.assert_array_equal(detection.xyxy, copied.xyxy)
        assert (
            detection.segmentation_points == copied.segmentation_points
        )  # List comparison
        np.testing.assert_array_equal(detection.class_id, copied.class_id)
        assert detection.image_size == copied.image_size

        # Modify the copy and check that the original is unchanged
        copied.class_id[0] = 99
        assert detection.class_id[0] != 99

    def test_xywh_property(self, detection_text_file):
        """Test the xywh property."""
        image_size = (640, 480)
        detection = Detection.from_detection_text_path(detection_text_file, image_size)

        xywh = detection.xywh
        assert len(xywh) == 3

        # Expected xywh values from the test file
        expected_xywh = [
            (0.5, 0.5, 0.2, 0.3),
            (0.2, 0.2, 0.1, 0.1),
            (0.8, 0.8, 0.2, 0.1),
        ]

        # Check each value with a small tolerance for floating point differences
        for i in range(3):
            assert np.allclose(xywh[i], expected_xywh[i], rtol=1e-5)

    def test_string_representation(self, detection_text_file):
        """Test string representation of Detection objects."""
        image_size = (640, 480)
        detection = Detection.from_detection_text_path(detection_text_file, image_size)

        str_repr = str(detection)
        assert "Detection" in str_repr
        assert "num_detections=3" in str_repr
        assert "image_size=(640, 480)" in str_repr

        # Test repr too
        repr_str = repr(detection)
        assert repr_str == str_repr

    def test_get_index(self, detection_text_file):
        """Test getting a specific detection by index."""
        image_size = (640, 480)
        detection = Detection.from_detection_text_path(detection_text_file, image_size)

        # Get the second detection (index 1)
        single_detection = detection.get_index(1)

        # Check that it's a Detection object with only one item
        assert isinstance(single_detection, Detection)
        assert len(single_detection.xyxy) == 1
        assert len(single_detection.segmentation_points) == 1
        assert len(single_detection.class_id) == 1
        assert single_detection.image_size == image_size

        # Check that it contains the correct data
        assert single_detection.class_id[0] == detection.class_id[1]
        np.testing.assert_array_equal(single_detection.xyxy[0], detection.xyxy[1])
        assert (
            single_detection.segmentation_points[0] == detection.segmentation_points[1]
        )

        # Test that modifying the extracted detection doesn't affect the original
        single_detection.class_id[0] = 99
        assert detection.class_id[1] != 99
