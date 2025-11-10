"""Tests for position filter classes.

This module contains comprehensive pytest tests for EdgeProximityFilter and
CenterDetectionFilter, which filter detections based on position relative to
image edges and center.
"""

import numpy as np
import pytest
from PIL import Image

from action_labeler.detections.detection import Detection
from action_labeler.filters.position import CenterDetectionFilter, EdgeProximityFilter
from tests.filters.helpers import (
    assert_all_fail,
    assert_all_pass,
    assert_filter_validates_indices,
    count_passing_detections,
    get_failing_indices,
    get_passing_indices,
)


class TestEdgeProximityFilterConstructor:
    """Test cases for EdgeProximityFilter constructor validation."""

    def test_constructor_with_negative_min_distance_raises_error(self):
        """Test that negative min_distance_pixels raises ValueError.

        The min_distance_pixels parameter must be non-negative to ensure
        valid distance calculations.
        """
        with pytest.raises(ValueError) as excinfo:
            EdgeProximityFilter(min_distance_pixels=-1)

        assert "min_distance_pixels must be non-negative" in str(excinfo.value)

    def test_constructor_with_zero_min_distance(self):
        """Test that min_distance_pixels=0 is valid.

        Zero pixels is a valid threshold, meaning detections touching edges
        would be considered edge detections.
        """
        filter_obj = EdgeProximityFilter(min_distance_pixels=0)
        assert filter_obj.min_distance_pixels == 0
        assert filter_obj.include_edge_detections is False

    def test_constructor_with_default_parameters(self):
        """Test constructor with default parameters."""
        filter_obj = EdgeProximityFilter()
        assert filter_obj.min_distance_pixels == 5
        assert filter_obj.include_edge_detections is False

    def test_constructor_with_custom_min_distance(self):
        """Test constructor with custom min_distance_pixels value."""
        filter_obj = EdgeProximityFilter(min_distance_pixels=10)
        assert filter_obj.min_distance_pixels == 10
        assert filter_obj.include_edge_detections is False

    def test_constructor_with_include_edge_detections_true(self):
        """Test constructor with include_edge_detections=True."""
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=True
        )
        assert filter_obj.min_distance_pixels == 5
        assert filter_obj.include_edge_detections is True

    def test_constructor_with_large_min_distance(self):
        """Test constructor with very large min_distance_pixels value."""
        filter_obj = EdgeProximityFilter(min_distance_pixels=1000)
        assert filter_obj.min_distance_pixels == 1000
        assert filter_obj.include_edge_detections is False


class TestEdgeProximityFilterExcludeEdges:
    """Test EdgeProximityFilter with include_edge_detections=False (default)."""

    def test_exclude_edges_with_default_parameters(self, edge_detection):
        """Test default behavior excludes edge detections within 5px.

        edge_detection has:
        - Detection 0: Left edge (x1=0)
        - Detection 1: Top edge (y1=0)
        - Detection 2: Right edge (x2=640)
        - Detection 3: Bottom edge (y2=480)
        - Detection 4: Center (far from edges)

        With default min_distance_pixels=5, only detection 4 should pass.
        """
        filter_obj = EdgeProximityFilter()
        expected_valid_indices = [4]  # Only center detection
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_exclude_edges_with_min_distance_5px(self, edge_detection):
        """Test explicitly setting min_distance_pixels=5 to exclude edges.

        All edge detections (0-3) touch edges (distance=0), so they fail.
        Center detection (4) is far from edges, so it passes.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=False
        )
        expected_valid_indices = [4]
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_exclude_edges_with_min_distance_10px(self, edge_detection):
        """Test excluding detections within 10 pixels of any edge.

        All edge detections (0-3) are at or near edges, so they fail.
        Center detection (4) is far from edges (>200px), so it passes.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=10, include_edge_detections=False
        )
        expected_valid_indices = [4]
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_exclude_edges_with_min_distance_50px(self, edge_detection):
        """Test excluding detections within 50 pixels of any edge.

        All edge detections (0-3) are at edges (distance=0).
        Center detection (4) is at approximately (280-360, 200-280),
        which is >200px from all edges, so it passes.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=50, include_edge_detections=False
        )
        expected_valid_indices = [4]
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_exclude_edges_with_min_distance_100px(self, edge_detection):
        """Test excluding detections within 100 pixels of any edge.

        edge_detection fixture (640x480 image):
        - Detection 4: bbox [280, 200, 360, 280]
          - dist_left = 280
          - dist_top = 200
          - dist_right = 640 - 360 = 280
          - dist_bottom = 480 - 280 = 200
          - min_edge_distance = 200 >= 100, so it passes
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=100, include_edge_detections=False
        )
        expected_valid_indices = [4]
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_exclude_edges_with_zero_min_distance(self, edge_detection):
        """Test with min_distance_pixels=0 doesn't exclude any detections.

        With min_distance_pixels=0, the condition is: is_near_edge = (distance < 0)
        Since distances are always >= 0, no detections are considered edge detections.
        All detections pass when excluding edges.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=0, include_edge_detections=False
        )
        # All detections have distance >= 0, so none are considered edge detections
        assert_all_pass(filter_obj, edge_detection)

    def test_exclude_edges_with_very_large_min_distance(self, edge_detection):
        """Test with min_distance_pixels larger than center detection distance.

        edge_detection fixture (640x480 image):
        - Detection 4: min_edge_distance = 200 (distance to top/bottom)
        - With min_distance_pixels=300, even detection 4 fails
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=300, include_edge_detections=False
        )
        assert_all_fail(filter_obj, edge_detection)


class TestEdgeProximityFilterIncludeEdges:
    """Test EdgeProximityFilter with include_edge_detections=True."""

    def test_include_edges_with_min_distance_5px(self, edge_detection):
        """Test including only edge detections within 5 pixels.

        All edge detections (0-3) are at edges (distance=0), so they pass.
        Center detection (4) is far from edges, so it fails.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=True
        )
        expected_valid_indices = [0, 1, 2, 3]  # Only edge detections
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_include_edges_with_min_distance_10px(self, edge_detection):
        """Test including only edge detections within 10 pixels.

        All edge detections (0-3) touch edges (distance=0 < 10), so they pass.
        Center detection (4) is far from edges, so it fails.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=10, include_edge_detections=True
        )
        expected_valid_indices = [0, 1, 2, 3]
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_include_edges_with_min_distance_50px(self, edge_detection):
        """Test including only edge detections within 50 pixels.

        All edge detections (0-3) are at edges (distance=0 < 50).
        Center detection (4) has min_edge_distance=200 >= 50, so it fails.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=50, include_edge_detections=True
        )
        expected_valid_indices = [0, 1, 2, 3]
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_include_edges_with_min_distance_100px(self, edge_detection):
        """Test including only edge detections within 100 pixels.

        All edge detections (0-3) are at edges (distance=0 < 100).
        Center detection (4) has min_edge_distance=200 >= 100, so it fails.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=100, include_edge_detections=True
        )
        expected_valid_indices = [0, 1, 2, 3]
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_include_edges_with_zero_min_distance(self, edge_detection):
        """Test with min_distance_pixels=0 includes no detections.

        With min_distance_pixels=0, the condition is: is_near_edge = (distance < 0)
        Since distances are always >= 0, no detections are considered edge detections.
        No detections pass when including only edges.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=0, include_edge_detections=True
        )
        # All detections have distance >= 0, so none are considered edge detections
        assert_all_fail(filter_obj, edge_detection)

    def test_include_edges_with_very_large_min_distance(self, edge_detection):
        """Test with min_distance_pixels larger than center detection distance.

        edge_detection fixture (640x480 image):
        - All detections (0-4) have min_edge_distance < 300
        - With min_distance_pixels=300, all detections pass
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=300, include_edge_detections=True
        )
        assert_all_pass(filter_obj, edge_detection)


class TestEdgeProximityFilterSingleDetection:
    """Test EdgeProximityFilter with single_detection fixture."""

    def test_single_centered_detection_excluded_edges(self, single_detection):
        """Test that centered detection passes when excluding edges.

        single_detection has bbox [256, 192, 384, 288] on 640x480 image:
        - dist_left = 256
        - dist_top = 192
        - dist_right = 640 - 384 = 256
        - dist_bottom = 480 - 288 = 192
        - min_edge_distance = 192

        With min_distance_pixels=5, this is far from edges and passes.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=False
        )
        assert_all_pass(filter_obj, single_detection)

    def test_single_centered_detection_included_edges(self, single_detection):
        """Test that centered detection fails when including only edges.

        With include_edge_detections=True and min_distance_pixels=5,
        only edge detections pass. The centered detection fails.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=True
        )
        assert_all_fail(filter_obj, single_detection)

    def test_single_detection_with_large_threshold(self, single_detection):
        """Test single detection with threshold larger than distance to edge.

        single_detection has min_edge_distance=192.
        With min_distance_pixels=200, it's considered an edge detection.
        """
        # Exclude edges: should fail (distance 192 < 200)
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=200, include_edge_detections=False
        )
        assert_all_fail(filter_obj, single_detection)

        # Include edges: should pass (distance 192 < 200)
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=200, include_edge_detections=True
        )
        assert_all_pass(filter_obj, single_detection)


class TestEdgeProximityFilterEmptyDetection:
    """Test EdgeProximityFilter with empty_detection fixture."""

    def test_empty_detection_exclude_edges(self, empty_detection):
        """Test that empty detection works correctly when excluding edges.

        With no detections, the filter should not raise errors.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=False
        )
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []

    def test_empty_detection_include_edges(self, empty_detection):
        """Test that empty detection works correctly when including edges.

        With no detections, the filter should not raise errors.
        """
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=True
        )
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestEdgeProximityFilterBoundaryConditions:
    """Test boundary conditions for EdgeProximityFilter."""

    def test_detection_exactly_at_threshold_distance(self, sample_image):
        """Test detection exactly at the threshold distance from edge.

        Create a detection with min_edge_distance exactly equal to threshold.
        With min_distance_pixels=10, a detection at distance 10 should NOT
        be considered an edge detection (is_near_edge checks distance < threshold).
        """
        # Detection with x1=10 (dist_left=10)
        xyxy = np.array([[10, 200, 100, 300]])
        class_id = np.array([0])
        segmentation_points = [[0.0156, 0.417, 0.156, 0.417, 0.156, 0.625, 0.0156, 0.625]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Distance is exactly 10, which is NOT < 10, so not an edge detection
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=10, include_edge_detections=False
        )
        assert_all_pass(filter_obj, detection)  # Should pass (not near edge)

        # When including edges, should fail (distance 10 is NOT < 10)
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=10, include_edge_detections=True
        )
        assert_all_fail(filter_obj, detection)  # Should fail (not an edge detection)

    def test_detection_one_pixel_inside_threshold(self, sample_image):
        """Test detection one pixel inside the threshold distance.

        Create a detection with min_edge_distance = threshold - 1.
        """
        # Detection with x1=9 (dist_left=9)
        xyxy = np.array([[9, 200, 100, 300]])
        class_id = np.array([0])
        segmentation_points = [[0.0141, 0.417, 0.156, 0.417, 0.156, 0.625, 0.0141, 0.625]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Distance is 9, which is < 10, so it's an edge detection
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=10, include_edge_detections=False
        )
        assert_all_fail(filter_obj, detection)  # Should fail (near edge)

        # When including edges, should pass
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=10, include_edge_detections=True
        )
        assert_all_pass(filter_obj, detection)  # Should pass (is edge detection)


class TestEdgeProximityFilterAllEdges:
    """Test that EdgeProximityFilter checks all 4 edges."""

    def test_left_edge_detection(self, sample_image):
        """Test detection near left edge is correctly identified.

        Detection with x1=0 (touching left edge).
        """
        xyxy = np.array([[0, 200, 80, 280]])
        class_id = np.array([0])
        segmentation_points = [[0.0, 0.417, 0.125, 0.417, 0.125, 0.583, 0.0, 0.583]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Should be excluded when excluding edges
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=False
        )
        assert_all_fail(filter_obj, detection)

        # Should be included when including edges
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=True
        )
        assert_all_pass(filter_obj, detection)

    def test_top_edge_detection(self, sample_image):
        """Test detection near top edge is correctly identified.

        Detection with y1=0 (touching top edge).
        """
        xyxy = np.array([[280, 0, 360, 60]])
        class_id = np.array([0])
        segmentation_points = [[0.4375, 0.0, 0.5625, 0.0, 0.5625, 0.125, 0.4375, 0.125]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Should be excluded when excluding edges
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=False
        )
        assert_all_fail(filter_obj, detection)

        # Should be included when including edges
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=True
        )
        assert_all_pass(filter_obj, detection)

    def test_right_edge_detection(self, sample_image):
        """Test detection near right edge is correctly identified.

        Detection with x2=640 (touching right edge on 640x480 image).
        """
        xyxy = np.array([[560, 200, 640, 280]])
        class_id = np.array([0])
        segmentation_points = [[0.875, 0.417, 1.0, 0.417, 1.0, 0.583, 0.875, 0.583]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Should be excluded when excluding edges
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=False
        )
        assert_all_fail(filter_obj, detection)

        # Should be included when including edges
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=True
        )
        assert_all_pass(filter_obj, detection)

    def test_bottom_edge_detection(self, sample_image):
        """Test detection near bottom edge is correctly identified.

        Detection with y2=480 (touching bottom edge on 640x480 image).
        """
        xyxy = np.array([[280, 420, 360, 480]])
        class_id = np.array([0])
        segmentation_points = [[0.4375, 0.875, 0.5625, 0.875, 0.5625, 1.0, 0.4375, 1.0]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # Should be excluded when excluding edges
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=False
        )
        assert_all_fail(filter_obj, detection)

        # Should be included when including edges
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=5, include_edge_detections=True
        )
        assert_all_pass(filter_obj, detection)

    def test_minimum_distance_uses_closest_edge(self, sample_image):
        """Test that filter uses the minimum distance to any edge.

        Create a detection closer to left edge than other edges.
        The filter should use the minimum distance (left edge distance).
        """
        # Detection at x1=5 (dist_left=5), other edges are much farther
        xyxy = np.array([[5, 200, 100, 300]])
        class_id = np.array([0])
        segmentation_points = [[0.0078, 0.417, 0.156, 0.417, 0.156, 0.625, 0.0078, 0.625]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # With threshold=10, detection at dist=5 should be edge detection
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=10, include_edge_detections=False
        )
        assert_all_fail(filter_obj, detection)

        # With threshold=4, detection at dist=5 should NOT be edge detection
        filter_obj = EdgeProximityFilter(
            min_distance_pixels=4, include_edge_detections=False
        )
        assert_all_pass(filter_obj, detection)


class TestCenterDetectionFilterConstructor:
    """Test cases for CenterDetectionFilter constructor validation."""

    def test_constructor_with_invalid_region_raises_error(self):
        """Test that invalid region parameter raises ValueError.

        Only specific region values are allowed: "center", "edges",
        "top", "bottom", "left", "right".
        """
        with pytest.raises(ValueError) as excinfo:
            CenterDetectionFilter(region="invalid")

        assert "region must be one of" in str(excinfo.value)
        assert "invalid" in str(excinfo.value)

    def test_constructor_with_margin_less_than_zero_raises_error(self):
        """Test that margin < 0 raises ValueError.

        Margin must be between 0.0 and 0.5 (inclusive).
        """
        with pytest.raises(ValueError) as excinfo:
            CenterDetectionFilter(region="center", margin=-0.1)

        assert "margin must be between 0.0 and 0.5" in str(excinfo.value)

    def test_constructor_with_margin_greater_than_half_raises_error(self):
        """Test that margin > 0.5 raises ValueError.

        Margin must be between 0.0 and 0.5 (inclusive).
        """
        with pytest.raises(ValueError) as excinfo:
            CenterDetectionFilter(region="center", margin=0.6)

        assert "margin must be between 0.0 and 0.5" in str(excinfo.value)

    def test_constructor_with_margin_exactly_zero(self):
        """Test that margin=0.0 is valid."""
        filter_obj = CenterDetectionFilter(region="center", margin=0.0)
        assert filter_obj.region == "center"
        assert filter_obj.margin == 0.0

    def test_constructor_with_margin_exactly_half(self):
        """Test that margin=0.5 is valid."""
        filter_obj = CenterDetectionFilter(region="center", margin=0.5)
        assert filter_obj.region == "center"
        assert filter_obj.margin == 0.5

    def test_constructor_with_default_parameters(self):
        """Test constructor with default parameters."""
        filter_obj = CenterDetectionFilter()
        assert filter_obj.region == "center"
        assert filter_obj.margin == 0.3

    def test_constructor_with_all_valid_regions(self):
        """Test constructor accepts all valid region values."""
        valid_regions = ["center", "edges", "top", "bottom", "left", "right"]

        for region in valid_regions:
            filter_obj = CenterDetectionFilter(region=region, margin=0.3)
            assert filter_obj.region == region
            assert filter_obj.margin == 0.3


class TestCenterDetectionFilterCenterRegion:
    """Test CenterDetectionFilter with region="center"."""

    def test_center_region_with_default_margin(self, multi_class_detection):
        """Test center region with default margin (0.3).

        multi_class_detection on 640x480 image:
        - Detection 0: [256, 192, 384, 288] -> center (320, 240)
          - norm_x = 320/640 = 0.5, norm_y = 240/480 = 0.5
          - In center region: 0.2 <= 0.5 <= 0.8 and 0.2 <= 0.5 <= 0.8 ✓
        - Detection 1: [50, 50, 150, 100] -> center (100, 75)
          - norm_x = 100/640 = 0.156, norm_y = 75/480 = 0.156
          - NOT in center region
        - Detection 2: [480, 360, 600, 440] -> center (540, 400)
          - norm_x = 540/640 = 0.844, norm_y = 400/480 = 0.833
          - NOT in center region
        - Detection 3: [20, 200, 100, 280] -> center (60, 240)
          - norm_x = 60/640 = 0.094, norm_y = 240/480 = 0.5
          - NOT in center region (x out of range)
        - Detection 4: [540, 200, 620, 280] -> center (580, 240)
          - norm_x = 580/640 = 0.906, norm_y = 240/480 = 0.5
          - NOT in center region (x out of range)
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.3)
        expected_valid_indices = [0]  # Only detection 0 is centered
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_center_region_with_small_margin(self, single_detection):
        """Test center region with small margin (0.1).

        single_detection bbox [256, 192, 384, 288] on 640x480 image:
        - center: (320, 240)
        - norm_x = 0.5, norm_y = 0.5
        - Center region: 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6
        - Detection is exactly at center, so it passes
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.1)
        assert_all_pass(filter_obj, single_detection)

    def test_center_region_with_large_margin(self, multi_class_detection):
        """Test center region with large margin (0.4).

        With margin=0.4, center region is 0.1 <= x <= 0.9 and 0.1 <= y <= 0.9.
        This should include more detections.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.4)
        passing = get_passing_indices(filter_obj, multi_class_detection)
        # Should include more detections than with smaller margin
        assert len(passing) >= 1

    def test_center_region_with_zero_margin(self, single_detection):
        """Test center region with margin=0.0.

        With margin=0.0, only detections exactly at (0.5, 0.5) pass.
        single_detection is at exactly (0.5, 0.5), so it passes.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.0)
        assert_all_pass(filter_obj, single_detection)

    def test_center_region_with_max_margin(self, multi_class_detection):
        """Test center region with margin=0.5.

        With margin=0.5, center region covers entire image.
        All detections should pass.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.5)
        assert_all_pass(filter_obj, multi_class_detection)


class TestCenterDetectionFilterEdgesRegion:
    """Test CenterDetectionFilter with region="edges"."""

    def test_edges_region_with_default_margin(self, multi_class_detection):
        """Test edges region with default margin (0.3).

        Edges region is the complement of center region.
        With margin=0.3, edges are outside 0.2 <= x <= 0.8 and 0.2 <= y <= 0.8.

        multi_class_detection:
        - Detection 0: norm (0.5, 0.5) -> in center, NOT in edges
        - Detection 1: norm (0.156, 0.156) -> in edges ✓
        - Detection 2: norm (0.844, 0.833) -> in edges ✓
        - Detection 3: norm (0.094, 0.5) -> in edges ✓
        - Detection 4: norm (0.906, 0.5) -> in edges ✓
        """
        filter_obj = CenterDetectionFilter(region="edges", margin=0.3)
        expected_valid_indices = [1, 2, 3, 4]  # All except center
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_edges_region_with_small_margin(self, single_detection):
        """Test edges region with small margin (0.1).

        single_detection is at (0.5, 0.5) which is in center region.
        With edges region, it should fail.
        """
        filter_obj = CenterDetectionFilter(region="edges", margin=0.1)
        assert_all_fail(filter_obj, single_detection)

    def test_edges_region_with_zero_margin(self, multi_class_detection):
        """Test edges region with margin=0.0.

        With margin=0.0, only detections NOT exactly at (0.5, 0.5) are in edges.
        Detection 0 is at (0.5, 0.5), so only it fails.
        """
        filter_obj = CenterDetectionFilter(region="edges", margin=0.0)
        expected_valid_indices = [1, 2, 3, 4]
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_edges_region_with_max_margin(self, multi_class_detection):
        """Test edges region with margin=0.5.

        With margin=0.5, center region covers entire image.
        Edges region is empty, so no detections pass.
        """
        filter_obj = CenterDetectionFilter(region="edges", margin=0.5)
        assert_all_fail(filter_obj, multi_class_detection)


class TestCenterDetectionFilterTopRegion:
    """Test CenterDetectionFilter with region="top"."""

    def test_top_region_with_margin_half(self, multi_class_detection):
        """Test top region with margin=0.5 (top half of image).

        multi_class_detection on 640x480 image:
        - Detection 0: center_y=240 -> norm_y=0.5 <= 0.5 ✓
        - Detection 1: center_y=75 -> norm_y=0.156 <= 0.5 ✓
        - Detection 2: center_y=400 -> norm_y=0.833, NOT in top
        - Detection 3: center_y=240 -> norm_y=0.5 <= 0.5 ✓
        - Detection 4: center_y=240 -> norm_y=0.5 <= 0.5 ✓
        """
        filter_obj = CenterDetectionFilter(region="top", margin=0.5)
        expected_valid_indices = [0, 1, 3, 4]  # Detections in top half (norm_y <= 0.5)
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_top_region_with_small_margin(self, multi_class_detection):
        """Test top region with small margin (0.2).

        Only detections in top 20% of image (norm_y <= 0.2) should pass.
        Detection 1: norm_y = 0.156 <= 0.2 ✓
        """
        filter_obj = CenterDetectionFilter(region="top", margin=0.2)
        expected_valid_indices = [1]
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_top_region_with_centered_detection(self, single_detection):
        """Test top region with centered detection.

        single_detection is at norm_y=0.5, which IS in top half (0.5 <= 0.5).
        """
        filter_obj = CenterDetectionFilter(region="top", margin=0.5)
        assert_all_pass(filter_obj, single_detection)

    def test_top_region_boundary_condition(self, sample_image):
        """Test detection exactly at the top margin boundary.

        Create detection with norm_y exactly equal to margin.
        """
        # Detection with center_y = 0.3 * 480 = 144
        xyxy = np.array([[200, 124, 300, 164]])  # center_y = 144
        class_id = np.array([0])
        segmentation_points = [[0.312, 0.258, 0.469, 0.258, 0.469, 0.342, 0.312, 0.342]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # norm_y = 144/480 = 0.3
        # With margin=0.3, check if norm_y <= 0.3 (should pass)
        filter_obj = CenterDetectionFilter(region="top", margin=0.3)
        assert_all_pass(filter_obj, detection)


class TestCenterDetectionFilterBottomRegion:
    """Test CenterDetectionFilter with region="bottom"."""

    def test_bottom_region_with_margin_half(self, multi_class_detection):
        """Test bottom region with margin=0.5 (bottom half of image).

        multi_class_detection on 640x480 image:
        - Detection 0: norm_y=0.5 >= 0.5 ✓
        - Detection 1: norm_y=0.156, NOT in bottom
        - Detection 2: norm_y=0.833 >= 0.5 ✓
        - Detection 3: norm_y=0.5 >= 0.5 ✓
        - Detection 4: norm_y=0.5 >= 0.5 ✓
        """
        filter_obj = CenterDetectionFilter(region="bottom", margin=0.5)
        expected_valid_indices = [0, 2, 3, 4]  # Detections in bottom half (norm_y >= 0.5)
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_bottom_region_with_small_margin(self, multi_class_detection):
        """Test bottom region with small margin (0.2).

        Only detections in bottom 20% (norm_y >= 0.8) should pass.
        Detection 2: norm_y = 0.833 >= 0.8 ✓
        """
        filter_obj = CenterDetectionFilter(region="bottom", margin=0.2)
        expected_valid_indices = [2]
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_bottom_region_with_centered_detection(self, single_detection):
        """Test bottom region with centered detection.

        single_detection is at norm_y=0.5, which IS in bottom half (0.5 >= 0.5).
        """
        filter_obj = CenterDetectionFilter(region="bottom", margin=0.5)
        assert_all_pass(filter_obj, single_detection)

    def test_bottom_region_boundary_condition(self, sample_image):
        """Test detection exactly at the bottom margin boundary.

        Create detection with norm_y exactly equal to 1.0 - margin.
        """
        # Detection with center_y = 0.7 * 480 = 336
        xyxy = np.array([[200, 316, 300, 356]])  # center_y = 336
        class_id = np.array([0])
        segmentation_points = [[0.312, 0.658, 0.469, 0.658, 0.469, 0.742, 0.312, 0.742]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # norm_y = 336/480 = 0.7
        # With margin=0.3, check if norm_y >= 0.7 (should pass)
        filter_obj = CenterDetectionFilter(region="bottom", margin=0.3)
        assert_all_pass(filter_obj, detection)


class TestCenterDetectionFilterLeftRegion:
    """Test CenterDetectionFilter with region="left"."""

    def test_left_region_with_margin_half(self, multi_class_detection):
        """Test left region with margin=0.5 (left half of image).

        multi_class_detection on 640x480 image:
        - Detection 0: norm_x=0.5 <= 0.5 ✓
        - Detection 1: norm_x=0.156 <= 0.5 ✓
        - Detection 2: norm_x=0.844, NOT in left
        - Detection 3: norm_x=0.094 <= 0.5 ✓
        - Detection 4: norm_x=0.906, NOT in left
        """
        filter_obj = CenterDetectionFilter(region="left", margin=0.5)
        expected_valid_indices = [0, 1, 3]  # Detections in left half (norm_x <= 0.5)
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_left_region_with_small_margin(self, multi_class_detection):
        """Test left region with small margin (0.1).

        Only detections in left 10% (norm_x <= 0.1) should pass.
        Detection 3: norm_x = 0.094 <= 0.1 ✓
        """
        filter_obj = CenterDetectionFilter(region="left", margin=0.1)
        expected_valid_indices = [3]
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_left_region_with_centered_detection(self, single_detection):
        """Test left region with centered detection.

        single_detection is at norm_x=0.5, which IS in left half (0.5 <= 0.5).
        """
        filter_obj = CenterDetectionFilter(region="left", margin=0.5)
        assert_all_pass(filter_obj, single_detection)

    def test_left_region_boundary_condition(self, sample_image):
        """Test detection exactly at the left margin boundary.

        Create detection with norm_x exactly equal to margin.
        """
        # Detection with center_x = 0.3 * 640 = 192
        xyxy = np.array([[172, 200, 212, 300]])  # center_x = 192
        class_id = np.array([0])
        segmentation_points = [[0.269, 0.417, 0.331, 0.417, 0.331, 0.625, 0.269, 0.625]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # norm_x = 192/640 = 0.3
        # With margin=0.3, check if norm_x <= 0.3 (should pass)
        filter_obj = CenterDetectionFilter(region="left", margin=0.3)
        assert_all_pass(filter_obj, detection)


class TestCenterDetectionFilterRightRegion:
    """Test CenterDetectionFilter with region="right"."""

    def test_right_region_with_margin_half(self, multi_class_detection):
        """Test right region with margin=0.5 (right half of image).

        multi_class_detection on 640x480 image:
        - Detection 0: norm_x=0.5 >= 0.5 ✓
        - Detection 1: norm_x=0.156, NOT in right
        - Detection 2: norm_x=0.844 >= 0.5 ✓
        - Detection 3: norm_x=0.094, NOT in right
        - Detection 4: norm_x=0.906 >= 0.5 ✓
        """
        filter_obj = CenterDetectionFilter(region="right", margin=0.5)
        expected_valid_indices = [0, 2, 4]  # Detections in right half (norm_x >= 0.5)
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_right_region_with_small_margin(self, multi_class_detection):
        """Test right region with small margin (0.1).

        Only detections in right 10% (norm_x >= 0.9) should pass.
        Detection 4: norm_x = 0.906 >= 0.9 ✓
        """
        filter_obj = CenterDetectionFilter(region="right", margin=0.1)
        expected_valid_indices = [4]
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

    def test_right_region_with_centered_detection(self, single_detection):
        """Test right region with centered detection.

        single_detection is at norm_x=0.5, which IS in right half (0.5 >= 0.5).
        """
        filter_obj = CenterDetectionFilter(region="right", margin=0.5)
        assert_all_pass(filter_obj, single_detection)

    def test_right_region_boundary_condition(self, sample_image):
        """Test detection exactly at the right margin boundary.

        Create detection with norm_x exactly equal to 1.0 - margin.
        """
        # Detection with center_x = 0.7 * 640 = 448
        xyxy = np.array([[428, 200, 468, 300]])  # center_x = 448
        class_id = np.array([0])
        segmentation_points = [[0.669, 0.417, 0.731, 0.417, 0.731, 0.625, 0.669, 0.625]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # norm_x = 448/640 = 0.7
        # With margin=0.3, check if norm_x >= 0.7 (should pass)
        filter_obj = CenterDetectionFilter(region="right", margin=0.3)
        assert_all_pass(filter_obj, detection)


class TestCenterDetectionFilterDifferentMargins:
    """Test CenterDetectionFilter with various margin values."""

    def test_center_with_margin_0_1(self, multi_class_detection):
        """Test center region with margin=0.1.

        Center region: 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6
        Detection 0: (0.5, 0.5) ✓
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.1)
        count = count_passing_detections(filter_obj, multi_class_detection)
        assert count == 1

    def test_center_with_margin_0_2(self, multi_class_detection):
        """Test center region with margin=0.2.

        Center region: 0.3 <= x <= 0.7 and 0.3 <= y <= 0.7
        Detection 0: (0.5, 0.5) ✓
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.2)
        count = count_passing_detections(filter_obj, multi_class_detection)
        assert count >= 1

    def test_center_with_margin_0_4(self, multi_class_detection):
        """Test center region with margin=0.4.

        Center region: 0.1 <= x <= 0.9 and 0.1 <= y <= 0.9
        This is a very large center region, should include most detections.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.4)
        count = count_passing_detections(filter_obj, multi_class_detection)
        # Should include multiple detections
        assert count >= 1


class TestCenterDetectionFilterEmptyDetection:
    """Test CenterDetectionFilter with empty_detection fixture."""

    def test_empty_detection_center_region(self, empty_detection):
        """Test center region with empty detection.

        With no detections, the filter should not raise errors.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.3)
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []

    def test_empty_detection_edges_region(self, empty_detection):
        """Test edges region with empty detection.

        With no detections, the filter should not raise errors.
        """
        filter_obj = CenterDetectionFilter(region="edges", margin=0.3)
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []

    def test_empty_detection_top_region(self, empty_detection):
        """Test top region with empty detection.

        With no detections, the filter should not raise errors.
        """
        filter_obj = CenterDetectionFilter(region="top", margin=0.5)
        assert len(empty_detection.xyxy) == 0
        passing = get_passing_indices(filter_obj, empty_detection)
        assert passing == []


class TestCenterDetectionFilterEdgeCases:
    """Test edge cases and boundary conditions for CenterDetectionFilter."""

    def test_center_region_excludes_edges_correctly(self, edge_detection):
        """Test that center region correctly excludes edge detections.

        edge_detection has 4 edge detections and 1 center detection.
        With center region, only the center detection should pass.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.3)
        expected_valid_indices = [4]  # Only center detection
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_edges_region_excludes_center_correctly(self, edge_detection):
        """Test that edges region correctly excludes center detection.

        edge_detection has 4 edge detections and 1 center detection.
        With edges region, only the edge detections should pass.
        """
        filter_obj = CenterDetectionFilter(region="edges", margin=0.3)
        expected_valid_indices = [0, 1, 2, 3]  # Only edge detections
        assert_filter_validates_indices(filter_obj, edge_detection, expected_valid_indices)

    def test_filter_consistency_across_multiple_calls(self, multi_class_detection):
        """Test that filter returns consistent results across multiple calls.

        Calling is_valid multiple times for the same detection should
        return the same result.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.3)

        # Call is_valid multiple times for each index
        for index in range(len(multi_class_detection.xyxy)):
            result1 = filter_obj.is_valid(
                multi_class_detection.image, index, multi_class_detection
            )
            result2 = filter_obj.is_valid(
                multi_class_detection.image, index, multi_class_detection
            )
            result3 = filter_obj.is_valid(
                multi_class_detection.image, index, multi_class_detection
            )
            assert result1 == result2 == result3

    def test_normalized_coordinates_calculation(self, sample_image):
        """Test that normalized coordinates are calculated correctly.

        Verify that the filter correctly normalizes detection centers to [0, 1].
        """
        # Create detection at top-left corner
        xyxy = np.array([[0, 0, 100, 100]])  # center: (50, 50)
        class_id = np.array([0])
        segmentation_points = [[0.0, 0.0, 0.156, 0.0, 0.156, 0.208, 0.0, 0.208]]
        keypoints = np.array([])

        detection = Detection(
            xyxy=xyxy,
            segmentation_points=segmentation_points,
            keypoints=keypoints,
            class_id=class_id,
            image=sample_image,
        )

        # norm_x = 50/640 = 0.078, norm_y = 50/480 = 0.104
        # With margin=0.2, this should be in top region (norm_y <= 0.2)
        filter_obj = CenterDetectionFilter(region="top", margin=0.2)
        assert_all_pass(filter_obj, detection)

        # Should also be in left region (norm_x <= 0.2)
        filter_obj = CenterDetectionFilter(region="left", margin=0.2)
        assert_all_pass(filter_obj, detection)


class TestCenterDetectionFilterHelperFunctions:
    """Test CenterDetectionFilter using helper functions to verify integration."""

    def test_count_passing_detections(self, multi_class_detection):
        """Test counting how many detections pass the filter.

        multi_class_detection with center region should have 1 passing detection.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.3)
        count = count_passing_detections(filter_obj, multi_class_detection)
        assert count == 1

    def test_get_passing_indices(self, multi_class_detection):
        """Test getting list of indices that pass the filter.

        With edges region, should return indices of non-centered detections.
        """
        filter_obj = CenterDetectionFilter(region="edges", margin=0.3)
        passing = get_passing_indices(filter_obj, multi_class_detection)
        assert passing == [1, 2, 3, 4]

    def test_get_failing_indices(self, multi_class_detection):
        """Test getting list of indices that fail the filter.

        With center region, only detection 0 should pass, others fail.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.3)
        failing = get_failing_indices(filter_obj, multi_class_detection)
        assert failing == [1, 2, 3, 4]

    def test_assert_filter_validates_indices_helper(self, multi_class_detection):
        """Test that assert_filter_validates_indices works correctly with CenterDetectionFilter.

        This verifies the integration between CenterDetectionFilter and test helpers.
        """
        filter_obj = CenterDetectionFilter(region="center", margin=0.3)
        expected_valid_indices = [0]

        # This should not raise an assertion error
        assert_filter_validates_indices(
            filter_obj, multi_class_detection, expected_valid_indices
        )

        # This should raise an assertion error (incorrect expectation)
        with pytest.raises(AssertionError):
            assert_filter_validates_indices(
                filter_obj, multi_class_detection, [0, 1, 2]
            )
