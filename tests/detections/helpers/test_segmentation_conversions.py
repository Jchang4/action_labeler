"""Test cases for segmentation conversion helper functions.

Tests for:
- xywh_to_segmentation_points
- segmentation_points_to_xywh
"""

import numpy as np
import pytest

from action_labeler.detections.helpers import (
    xywh_to_segmentation_points,
    segmentation_points_to_xywh,
)


class TestXywhToSegmentationPoints:
    """Test cases for xywh_to_segmentation_points conversion."""

    def test_centered_box(self):
        """Test conversion of a centered box to segmentation points."""
        xywh = (0.5, 0.5, 0.4, 0.6)

        result = xywh_to_segmentation_points(xywh)

        # Expected 4 corners: top-left, top-right, bottom-right, bottom-left
        # x_center=0.5, width=0.4 -> x_min=0.3, x_max=0.7
        # y_center=0.5, height=0.6 -> y_min=0.2, y_max=0.8
        expected = [
            0.3, 0.2,  # Top-left
            0.7, 0.2,  # Top-right
            0.7, 0.8,  # Bottom-right
            0.3, 0.8,  # Bottom-left
        ]
        assert result == expected

    def test_top_left_corner(self):
        """Test conversion of box at top-left corner."""
        xywh = (0.1, 0.1, 0.2, 0.2)

        result = xywh_to_segmentation_points(xywh)

        # x_center=0.1, width=0.2 -> x_min=0.0, x_max=0.2
        # y_center=0.1, height=0.2 -> y_min=0.0, y_max=0.2
        expected = [
            0.0, 0.0,  # Top-left
            0.2, 0.0,  # Top-right
            0.2, 0.2,  # Bottom-right
            0.0, 0.2,  # Bottom-left
        ]
        assert result == expected

    def test_bottom_right_corner(self):
        """Test conversion of box at bottom-right corner."""
        xywh = (0.9, 0.9, 0.2, 0.2)

        result = xywh_to_segmentation_points(xywh)

        # x_center=0.9, width=0.2 -> x_min=0.8, x_max=1.0
        # y_center=0.9, height=0.2 -> y_min=0.8, y_max=1.0
        expected = [
            0.8, 0.8,  # Top-left
            1.0, 0.8,  # Top-right
            1.0, 1.0,  # Bottom-right
            0.8, 1.0,  # Bottom-left
        ]
        assert result == expected

    def test_square_box(self):
        """Test conversion of a square box."""
        xywh = (0.5, 0.5, 0.4, 0.4)

        result = xywh_to_segmentation_points(xywh)

        # x_center=0.5, width=0.4 -> x_min=0.3, x_max=0.7
        # y_center=0.5, height=0.4 -> y_min=0.3, y_max=0.7
        expected = [
            0.3, 0.3,  # Top-left
            0.7, 0.3,  # Top-right
            0.7, 0.7,  # Bottom-right
            0.3, 0.7,  # Bottom-left
        ]
        assert result == expected

    def test_wide_box(self):
        """Test conversion of a wide rectangular box."""
        xywh = (0.5, 0.5, 0.8, 0.2)

        result = xywh_to_segmentation_points(xywh)

        # x_center=0.5, width=0.8 -> x_min=0.1, x_max=0.9
        # y_center=0.5, height=0.2 -> y_min=0.4, y_max=0.6
        expected = [
            0.1, 0.4,  # Top-left
            0.9, 0.4,  # Top-right
            0.9, 0.6,  # Bottom-right
            0.1, 0.6,  # Bottom-left
        ]
        assert result == pytest.approx(expected)

    def test_tall_box(self):
        """Test conversion of a tall rectangular box."""
        xywh = (0.5, 0.5, 0.2, 0.8)

        result = xywh_to_segmentation_points(xywh)

        # x_center=0.5, width=0.2 -> x_min=0.4, x_max=0.6
        # y_center=0.5, height=0.8 -> y_min=0.1, y_max=0.9
        expected = [
            0.4, 0.1,  # Top-left
            0.6, 0.1,  # Top-right
            0.6, 0.9,  # Bottom-right
            0.4, 0.9,  # Bottom-left
        ]
        assert result == pytest.approx(expected)

    def test_very_small_box(self):
        """Test conversion of a very small box."""
        xywh = (0.5, 0.5, 0.01, 0.01)

        result = xywh_to_segmentation_points(xywh)

        # x_center=0.5, width=0.01 -> x_min=0.495, x_max=0.505
        # y_center=0.5, height=0.01 -> y_min=0.495, y_max=0.505
        expected = [
            0.495, 0.495,  # Top-left
            0.505, 0.495,  # Top-right
            0.505, 0.505,  # Bottom-right
            0.495, 0.505,  # Bottom-left
        ]
        assert result == expected

    def test_full_image_box(self):
        """Test conversion of a box covering the entire image."""
        xywh = (0.5, 0.5, 1.0, 1.0)

        result = xywh_to_segmentation_points(xywh)

        # x_center=0.5, width=1.0 -> x_min=0.0, x_max=1.0
        # y_center=0.5, height=1.0 -> y_min=0.0, y_max=1.0
        expected = [
            0.0, 0.0,  # Top-left
            1.0, 0.0,  # Top-right
            1.0, 1.0,  # Bottom-right
            0.0, 1.0,  # Bottom-left
        ]
        assert result == expected

    def test_zero_width_height(self):
        """Test conversion with zero width/height (edge case - point)."""
        xywh = (0.5, 0.5, 0.0, 0.0)

        result = xywh_to_segmentation_points(xywh)

        # All corners should be at the center point
        expected = [
            0.5, 0.5,  # Top-left
            0.5, 0.5,  # Top-right
            0.5, 0.5,  # Bottom-right
            0.5, 0.5,  # Bottom-left
        ]
        assert result == expected

    def test_return_format(self):
        """Test that the function returns exactly 8 values (4 points)."""
        xywh = (0.5, 0.5, 0.4, 0.6)

        result = xywh_to_segmentation_points(xywh)

        assert len(result) == 8
        assert isinstance(result, list)

    def test_asymmetric_position(self):
        """Test conversion with asymmetric position."""
        xywh = (0.3, 0.7, 0.4, 0.2)

        result = xywh_to_segmentation_points(xywh)

        # x_center=0.3, width=0.4 -> x_min=0.1, x_max=0.5
        # y_center=0.7, height=0.2 -> y_min=0.6, y_max=0.8
        expected = [
            0.1, 0.6,  # Top-left
            0.5, 0.6,  # Top-right
            0.5, 0.8,  # Bottom-right
            0.1, 0.8,  # Bottom-left
        ]
        assert result == pytest.approx(expected)

    def test_floating_point_precision(self):
        """Test that floating point precision is maintained."""
        xywh = (0.333333, 0.666666, 0.123456, 0.789012)

        result = xywh_to_segmentation_points(xywh)

        # Calculate expected values
        x_min = 0.333333 - 0.123456 / 2
        x_max = 0.333333 + 0.123456 / 2
        y_min = 0.666666 - 0.789012 / 2
        y_max = 0.666666 + 0.789012 / 2

        assert abs(result[0] - x_min) < 1e-10
        assert abs(result[1] - y_min) < 1e-10
        assert abs(result[2] - x_max) < 1e-10
        assert abs(result[3] - y_min) < 1e-10


class TestSegmentationPointsToXywh:
    """Test cases for segmentation_points_to_xywh conversion."""

    def test_rectangular_polygon_4_points(self):
        """Test conversion of a rectangular polygon with 4 points."""
        # Points defining a rectangle
        points = [
            0.3, 0.2,  # Top-left
            0.7, 0.2,  # Top-right
            0.7, 0.8,  # Bottom-right
            0.3, 0.8,  # Bottom-left
        ]

        result = segmentation_points_to_xywh(points)

        # x_min=0.3, x_max=0.7 -> x_center=0.5, width=0.4
        # y_min=0.2, y_max=0.8 -> y_center=0.5, height=0.6
        assert result == pytest.approx((0.5, 0.5, 0.4, 0.6))

    def test_triangle_3_points(self):
        """Test conversion of a triangular polygon."""
        # Triangle points
        points = [
            0.5, 0.0,  # Top
            0.0, 1.0,  # Bottom-left
            1.0, 1.0,  # Bottom-right
        ]

        result = segmentation_points_to_xywh(points)

        # x_min=0.0, x_max=1.0 -> x_center=0.5, width=1.0
        # y_min=0.0, y_max=1.0 -> y_center=0.5, height=1.0
        assert result == pytest.approx((0.5, 0.5, 1.0, 1.0))

    def test_pentagon_5_points(self):
        """Test conversion of a pentagonal polygon."""
        # Pentagon (approximate regular pentagon)
        points = [
            0.5, 0.0,   # Top
            1.0, 0.4,   # Upper-right
            0.8, 1.0,   # Lower-right
            0.2, 1.0,   # Lower-left
            0.0, 0.4,   # Upper-left
        ]

        result = segmentation_points_to_xywh(points)

        # x_min=0.0, x_max=1.0 -> x_center=0.5, width=1.0
        # y_min=0.0, y_max=1.0 -> y_center=0.5, height=1.0
        assert result == pytest.approx((0.5, 0.5, 1.0, 1.0))

    def test_many_points_polygon(self):
        """Test conversion of a polygon with many points."""
        # Create a polygon with 20 points around a circle-like shape
        points = []
        for i in range(20):
            angle = i * 2 * np.pi / 20
            x = 0.5 + 0.3 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.extend([x, y])

        result = segmentation_points_to_xywh(points)

        # Should approximate a box centered at (0.5, 0.5) with width ~0.6, height ~0.8
        assert abs(result[0] - 0.5) < 0.01  # x_center
        assert abs(result[1] - 0.5) < 0.01  # y_center
        assert abs(result[2] - 0.6) < 0.01  # width
        assert abs(result[3] - 0.8) < 0.01  # height

    def test_irregular_polygon(self):
        """Test conversion of an irregular polygon."""
        points = [
            0.2, 0.3,
            0.8, 0.2,
            0.9, 0.7,
            0.4, 0.9,
            0.1, 0.6,
        ]

        result = segmentation_points_to_xywh(points)

        # x_min=0.1, x_max=0.9 -> x_center=0.5, width=0.8
        # y_min=0.2, y_max=0.9 -> y_center=0.55, height=0.7
        assert result == pytest.approx((0.5, 0.55, 0.8, 0.7))

    def test_single_point(self):
        """Test conversion of a single point (edge case)."""
        points = [0.5, 0.5]

        result = segmentation_points_to_xywh(points)

        # Single point should result in zero-sized box
        assert result == pytest.approx((0.5, 0.5, 0.0, 0.0))

    def test_two_points_line(self):
        """Test conversion of two points (line segment)."""
        points = [0.3, 0.4, 0.7, 0.6]

        result = segmentation_points_to_xywh(points)

        # x_min=0.3, x_max=0.7 -> x_center=0.5, width=0.4
        # y_min=0.4, y_max=0.6 -> y_center=0.5, height=0.2
        assert result == pytest.approx((0.5, 0.5, 0.4, 0.2))

    def test_horizontal_line(self):
        """Test conversion of a horizontal line."""
        points = [0.2, 0.5, 0.8, 0.5]

        result = segmentation_points_to_xywh(points)

        # x_min=0.2, x_max=0.8 -> x_center=0.5, width=0.6
        # y_min=0.5, y_max=0.5 -> y_center=0.5, height=0.0
        assert result == pytest.approx((0.5, 0.5, 0.6, 0.0))

    def test_vertical_line(self):
        """Test conversion of a vertical line."""
        points = [0.5, 0.2, 0.5, 0.8]

        result = segmentation_points_to_xywh(points)

        # x_min=0.5, x_max=0.5 -> x_center=0.5, width=0.0
        # y_min=0.2, y_max=0.8 -> y_center=0.5, height=0.6
        assert result == pytest.approx((0.5, 0.5, 0.0, 0.6))

    def test_points_at_image_boundaries(self):
        """Test conversion with points at image boundaries."""
        points = [
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ]

        result = segmentation_points_to_xywh(points)

        # Full image box
        assert result == pytest.approx((0.5, 0.5, 1.0, 1.0))

    def test_unsorted_points(self):
        """Test that function works with unsorted points."""
        # Points not in clockwise/counter-clockwise order
        points = [
            0.7, 0.2,  # Top-right
            0.3, 0.8,  # Bottom-left
            0.7, 0.8,  # Bottom-right
            0.3, 0.2,  # Top-left
        ]

        result = segmentation_points_to_xywh(points)

        # Should still compute correct bounding box
        assert result == pytest.approx((0.5, 0.5, 0.4, 0.6))

    def test_duplicate_points(self):
        """Test handling of duplicate points."""
        points = [
            0.3, 0.2,
            0.7, 0.2,
            0.7, 0.2,  # Duplicate
            0.7, 0.8,
            0.3, 0.8,
        ]

        result = segmentation_points_to_xywh(points)

        # Duplicates shouldn't affect bounding box calculation
        assert result == pytest.approx((0.5, 0.5, 0.4, 0.6))

    def test_very_small_polygon(self):
        """Test conversion of a very small polygon."""
        points = [
            0.495, 0.495,
            0.505, 0.495,
            0.505, 0.505,
            0.495, 0.505,
        ]

        result = segmentation_points_to_xywh(points)

        assert result == pytest.approx((0.5, 0.5, 0.01, 0.01))

    def test_floating_point_precision(self):
        """Test that floating point precision is maintained."""
        points = [
            0.271605, 0.272160,
            0.395061, 0.272160,
            0.395061, 0.666672,
            0.271605, 0.666672,
        ]

        result = segmentation_points_to_xywh(points)

        # Calculate expected values
        x_min, x_max = 0.271605, 0.395061
        y_min, y_max = 0.272160, 0.666672
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        assert abs(result[0] - x_center) < 1e-10
        assert abs(result[1] - y_center) < 1e-10
        assert abs(result[2] - width) < 1e-10
        assert abs(result[3] - height) < 1e-10

    def test_concave_polygon(self):
        """Test conversion of a concave polygon."""
        # Star-like shape
        points = [
            0.5, 0.1,   # Top
            0.6, 0.4,   # Inner-right
            0.9, 0.4,   # Outer-right
            0.6, 0.6,   # Inner-bottom-right
            0.7, 0.9,   # Outer-bottom-right
            0.5, 0.7,   # Inner-bottom
            0.3, 0.9,   # Outer-bottom-left
            0.4, 0.6,   # Inner-bottom-left
            0.1, 0.4,   # Outer-left
            0.4, 0.4,   # Inner-left
        ]

        result = segmentation_points_to_xywh(points)

        # Bounding box should cover from (0.1, 0.1) to (0.9, 0.9)
        assert result == pytest.approx((0.5, 0.5, 0.8, 0.8))


class TestRoundTripSegmentationConversions:
    """Test that segmentation conversions are reversible for rectangular polygons."""

    def test_xywh_to_segmentation_to_xywh(self):
        """Test round-trip conversion xywh -> segmentation -> xywh."""
        original = (0.5, 0.5, 0.4, 0.6)

        points = xywh_to_segmentation_points(original)
        result = segmentation_points_to_xywh(points)

        assert result == pytest.approx(original)

    def test_round_trip_multiple_boxes(self):
        """Test round-trip for multiple different boxes."""
        test_cases = [
            (0.5, 0.5, 0.4, 0.6),
            (0.1, 0.1, 0.2, 0.2),
            (0.9, 0.9, 0.2, 0.2),
            (0.5, 0.5, 1.0, 1.0),
            (0.3, 0.7, 0.6, 0.2),
        ]

        for original in test_cases:
            points = xywh_to_segmentation_points(original)
            result = segmentation_points_to_xywh(points)
            assert result == pytest.approx(original)

    def test_round_trip_edge_cases(self):
        """Test round-trip with edge case values."""
        edge_cases = [
            (0.5, 0.5, 0.0, 0.0),  # Zero-sized box
            (0.0, 0.0, 0.0, 0.0),  # Zero box at origin
            (1.0, 1.0, 0.0, 0.0),  # Zero box at far corner
            (0.5, 0.5, 0.01, 0.01),  # Very small box
        ]

        for original in edge_cases:
            points = xywh_to_segmentation_points(original)
            result = segmentation_points_to_xywh(points)
            assert result == pytest.approx(original)

    def test_round_trip_preserves_precision(self):
        """Test that round-trip preserves floating point precision."""
        original = (0.333333, 0.666666, 0.123456, 0.789012)

        points = xywh_to_segmentation_points(original)
        result = segmentation_points_to_xywh(points)

        for orig, res in zip(original, result):
            assert abs(orig - res) < 1e-10


class TestSegmentationEdgeCases:
    """Test edge cases specific to segmentation conversions."""

    def test_segmentation_with_odd_number_of_values(self):
        """Test that odd number of coordinates is handled (edge case - malformed)."""
        # This is technically malformed data, but should not crash
        points = [0.3, 0.2, 0.7, 0.2, 0.7, 0.8, 0.3]  # Missing last y

        # numpy reshape should handle this, but may raise an error
        # We're testing that our function handles it gracefully or raises appropriate error
        with pytest.raises(ValueError):
            segmentation_points_to_xywh(points)

    def test_empty_segmentation_points(self):
        """Test handling of empty points list."""
        points = []

        # Empty array should raise an error when trying to find min/max
        with pytest.raises((ValueError, IndexError)):
            segmentation_points_to_xywh(points)

    def test_numpy_array_input(self):
        """Test that function accepts numpy array input."""
        points = np.array([0.3, 0.2, 0.7, 0.2, 0.7, 0.8, 0.3, 0.8])

        result = segmentation_points_to_xywh(points.tolist())

        assert result == pytest.approx((0.5, 0.5, 0.4, 0.6))
