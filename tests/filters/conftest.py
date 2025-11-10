"""Shared fixtures for filter tests.

This module provides reusable pytest fixtures for testing filters.
Fixtures include Detection objects with various configurations to test
different filter scenarios.
"""

import numpy as np
import pytest
from PIL import Image

from action_labeler.detections.detection import Detection


@pytest.fixture
def sample_image():
    """Create a sample 640x480 RGB image for testing."""
    return Image.new("RGB", (640, 480))


@pytest.fixture
def small_image():
    """Create a small 100x100 RGB image for testing edge cases."""
    return Image.new("RGB", (100, 100))


@pytest.fixture
def large_image():
    """Create a large 1920x1080 RGB image for testing."""
    return Image.new("RGB", (1920, 1080))


@pytest.fixture
def empty_detection(sample_image):
    """Create an empty Detection with no detections."""
    return Detection.empty(image=sample_image)


@pytest.fixture
def single_detection(sample_image):
    """Create a Detection with a single detection at the center.

    Detection properties:
    - Class ID: 0
    - Center: (320, 240) - image center
    - Size: 128x96 pixels (20% of image width, 20% of height)
    - Bbox: [256, 192, 384, 288]
    """
    xyxy = np.array([[256, 192, 384, 288]])
    class_id = np.array([0])
    segmentation_points = [[0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6]]
    keypoints = np.array([])

    return Detection(
        xyxy=xyxy,
        segmentation_points=segmentation_points,
        keypoints=keypoints,
        class_id=class_id,
        image=sample_image,
    )


@pytest.fixture
def multi_class_detection(sample_image):
    """Create a Detection with multiple detections of different classes.

    Detections:
    - Detection 0: Class 0, center area
    - Detection 1: Class 1, top-left area
    - Detection 2: Class 2, bottom-right area
    - Detection 3: Class 0, left side
    - Detection 4: Class 1, right side
    """
    xyxy = np.array(
        [
            [256, 192, 384, 288],  # Class 0 - center
            [50, 50, 150, 100],  # Class 1 - top-left
            [480, 360, 600, 440],  # Class 2 - bottom-right
            [20, 200, 100, 280],  # Class 0 - left
            [540, 200, 620, 280],  # Class 1 - right
        ]
    )
    class_id = np.array([0, 1, 2, 0, 1])
    segmentation_points = [
        [0.4, 0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6],  # Detection 0
        [0.078, 0.104, 0.234, 0.104, 0.234, 0.208, 0.078, 0.208],  # Detection 1
        [0.75, 0.75, 0.9375, 0.75, 0.9375, 0.917, 0.75, 0.917],  # Detection 2
        [0.031, 0.417, 0.156, 0.417, 0.156, 0.583, 0.031, 0.583],  # Detection 3
        [0.844, 0.417, 0.969, 0.417, 0.969, 0.583, 0.844, 0.583],  # Detection 4
    ]
    keypoints = np.array([])

    return Detection(
        xyxy=xyxy,
        segmentation_points=segmentation_points,
        keypoints=keypoints,
        class_id=class_id,
        image=sample_image,
    )


@pytest.fixture
def varying_size_detection(sample_image):
    """Create a Detection with detections of varying sizes.

    Detections ordered from largest to smallest:
    - Detection 0: Large (200x150 pixels = 30000 px²)
    - Detection 1: Medium (100x80 pixels = 8000 px²)
    - Detection 2: Small (50x40 pixels = 2000 px²)
    - Detection 3: Tiny (20x20 pixels = 400 px²)
    """
    xyxy = np.array(
        [
            [220, 165, 420, 315],  # Large: 200x150
            [100, 100, 200, 180],  # Medium: 100x80
            [450, 300, 500, 340],  # Small: 50x40
            [550, 400, 570, 420],  # Tiny: 20x20
        ]
    )
    class_id = np.array([0, 0, 0, 0])
    segmentation_points = [
        [0.344, 0.344, 0.656, 0.344, 0.656, 0.656, 0.344, 0.656],  # Large
        [0.156, 0.208, 0.312, 0.208, 0.312, 0.375, 0.156, 0.375],  # Medium
        [0.703, 0.625, 0.781, 0.625, 0.781, 0.708, 0.703, 0.708],  # Small
        [0.859, 0.833, 0.891, 0.833, 0.891, 0.875, 0.859, 0.875],  # Tiny
    ]
    keypoints = np.array([])

    return Detection(
        xyxy=xyxy,
        segmentation_points=segmentation_points,
        keypoints=keypoints,
        class_id=class_id,
        image=sample_image,
    )


@pytest.fixture
def edge_detection(sample_image):
    """Create a Detection with detections near image edges.

    Detections:
    - Detection 0: Touching left edge (x1=0)
    - Detection 1: Touching top edge (y1=0)
    - Detection 2: Touching right edge (x2=640)
    - Detection 3: Touching bottom edge (y2=480)
    - Detection 4: Center (not near any edge)
    """
    xyxy = np.array(
        [
            [0, 200, 80, 280],  # Left edge
            [280, 0, 360, 60],  # Top edge
            [560, 200, 640, 280],  # Right edge
            [280, 420, 360, 480],  # Bottom edge
            [280, 200, 360, 280],  # Center (far from edges)
        ]
    )
    class_id = np.array([0, 0, 0, 0, 0])
    segmentation_points = [
        [0.0, 0.417, 0.125, 0.417, 0.125, 0.583, 0.0, 0.583],
        [0.4375, 0.0, 0.5625, 0.0, 0.5625, 0.125, 0.4375, 0.125],
        [0.875, 0.417, 1.0, 0.417, 1.0, 0.583, 0.875, 0.583],
        [0.4375, 0.875, 0.5625, 0.875, 0.5625, 1.0, 0.4375, 1.0],
        [0.4375, 0.417, 0.5625, 0.417, 0.5625, 0.583, 0.4375, 0.583],
    ]
    keypoints = np.array([])

    return Detection(
        xyxy=xyxy,
        segmentation_points=segmentation_points,
        keypoints=keypoints,
        class_id=class_id,
        image=sample_image,
    )


@pytest.fixture
def overlapping_detection(sample_image):
    """Create a Detection with overlapping detections.

    Detections:
    - Detection 0: Center box [240, 180, 400, 300]
    - Detection 1: Overlaps with 0 (high IoU ~0.4)
    - Detection 2: Isolated, no overlap
    """
    xyxy = np.array(
        [
            [240, 180, 400, 300],  # Center box
            [300, 220, 460, 340],  # Overlaps with 0
            [500, 50, 600, 150],  # Isolated
        ]
    )
    class_id = np.array([0, 0, 0])
    segmentation_points = [
        [0.375, 0.375, 0.625, 0.375, 0.625, 0.625, 0.375, 0.625],
        [0.469, 0.458, 0.719, 0.458, 0.719, 0.708, 0.469, 0.708],
        [0.781, 0.104, 0.938, 0.104, 0.938, 0.312, 0.781, 0.312],
    ]
    keypoints = np.array([])

    return Detection(
        xyxy=xyxy,
        segmentation_points=segmentation_points,
        keypoints=keypoints,
        class_id=class_id,
        image=sample_image,
    )


@pytest.fixture
def aspect_ratio_detection(sample_image):
    """Create a Detection with various aspect ratios.

    Detections:
    - Detection 0: Square (aspect ratio = 1.0)
    - Detection 1: Wide (aspect ratio = 4.0)
    - Detection 2: Tall (aspect ratio = 0.25)
    - Detection 3: Very wide (aspect ratio = 8.0)
    """
    xyxy = np.array(
        [
            [100, 100, 200, 200],  # Square: 100x100, ratio=1.0
            [250, 150, 450, 200],  # Wide: 200x50, ratio=4.0
            [500, 100, 525, 200],  # Tall: 25x100, ratio=0.25
            [100, 300, 420, 340],  # Very wide: 320x40, ratio=8.0
        ]
    )
    class_id = np.array([0, 0, 0, 0])
    segmentation_points = [
        [0.156, 0.208, 0.312, 0.208, 0.312, 0.417, 0.156, 0.417],
        [0.391, 0.312, 0.703, 0.312, 0.703, 0.417, 0.391, 0.417],
        [0.781, 0.208, 0.820, 0.208, 0.820, 0.417, 0.781, 0.417],
        [0.156, 0.625, 0.656, 0.625, 0.656, 0.708, 0.156, 0.708],
    ]
    keypoints = np.array([])

    return Detection(
        xyxy=xyxy,
        segmentation_points=segmentation_points,
        keypoints=keypoints,
        class_id=class_id,
        image=sample_image,
    )
