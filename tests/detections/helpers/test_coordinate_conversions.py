"""Test cases for coordinate conversion helper functions.

Tests for:
- xywh_to_xyxy
- xywhs_to_xyxys
- xyxy_to_xywh
- xyxys_to_xywhs
"""

import numpy as np
import pytest

from action_labeler.detections.helpers import (
    xywh_to_xyxy,
    xywhs_to_xyxys,
    xyxy_to_xywh,
    xyxys_to_xywhs,
)


class TestXywhToXyxy:
    """Test cases for xywh_to_xyxy conversion."""

    def test_centered_box(self):
        """Test conversion of a centered box."""
        xywh = (0.5, 0.5, 0.4, 0.6)
        image_size = (100, 100)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.5, width=0.4 -> x1=0.3*100=30, x2=0.7*100=70
        # y_center=0.5, height=0.6 -> y1=0.2*100=20, y2=0.8*100=80
        assert result == (30.0, 20.0, 70.0, 80.0)

    def test_top_left_corner(self):
        """Test conversion of box at top-left corner."""
        xywh = (0.1, 0.1, 0.2, 0.2)
        image_size = (100, 100)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.1, width=0.2 -> x1=0.0*100=0, x2=0.2*100=20
        # y_center=0.1, height=0.2 -> y1=0.0*100=0, y2=0.2*100=20
        assert result == (0.0, 0.0, 20.0, 20.0)

    def test_bottom_right_corner(self):
        """Test conversion of box at bottom-right corner."""
        xywh = (0.9, 0.9, 0.2, 0.2)
        image_size = (100, 100)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.9, width=0.2 -> x1=0.8*100=80, x2=1.0*100=100
        # y_center=0.9, height=0.2 -> y1=0.8*100=80, y2=1.0*100=100
        assert result == (80.0, 80.0, 100.0, 100.0)

    def test_rectangular_image_wide(self):
        """Test conversion with wide rectangular image."""
        xywh = (0.5, 0.5, 0.4, 0.2)
        image_size = (200, 100)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.5, width=0.4 -> x1=0.3*200=60, x2=0.7*200=140
        # y_center=0.5, height=0.2 -> y1=0.4*100=40, y2=0.6*100=60
        assert result == (60.0, 40.0, 140.0, 60.0)

    def test_rectangular_image_tall(self):
        """Test conversion with tall rectangular image."""
        xywh = (0.5, 0.5, 0.2, 0.4)
        image_size = (100, 200)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.5, width=0.2 -> x1=0.4*100=40, x2=0.6*100=60
        # y_center=0.5, height=0.4 -> y1=0.3*200=60, y2=0.7*200=140
        assert result == (40.0, 60.0, 60.0, 140.0)

    def test_very_small_box(self):
        """Test conversion of very small box."""
        xywh = (0.5, 0.5, 0.01, 0.01)
        image_size = (1000, 1000)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.5, width=0.01 -> x1=0.495*1000=495, x2=0.505*1000=505
        # y_center=0.5, height=0.01 -> y1=0.495*1000=495, y2=0.505*1000=505
        assert result == (495.0, 495.0, 505.0, 505.0)

    def test_very_large_box(self):
        """Test conversion of very large box."""
        xywh = (0.5, 0.5, 1.0, 1.0)
        image_size = (100, 100)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.5, width=1.0 -> x1=0.0*100=0, x2=1.0*100=100
        # y_center=0.5, height=1.0 -> y1=0.0*100=0, y2=1.0*100=100
        assert result == (0.0, 0.0, 100.0, 100.0)

    def test_zero_width_height(self):
        """Test conversion with zero width/height (edge case)."""
        xywh = (0.5, 0.5, 0.0, 0.0)
        image_size = (100, 100)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.5, width=0.0 -> x1=x2=50
        # y_center=0.5, height=0.0 -> y1=y2=50
        assert result == (50.0, 50.0, 50.0, 50.0)

    def test_large_image_dimensions(self):
        """Test conversion with large image dimensions."""
        xywh = (0.5, 0.5, 0.2, 0.3)
        image_size = (1920, 1080)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.5, width=0.2 -> x1=0.4*1920=768, x2=0.6*1920=1152
        # y_center=0.5, height=0.3 -> y1=0.35*1080=378, y2=0.65*1080=702
        assert result == (768.0, 378.0, 1152.0, 702.0)

    def test_4k_resolution(self):
        """Test conversion with 4K resolution."""
        xywh = (0.5, 0.5, 0.1, 0.1)
        image_size = (3840, 2160)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.5, width=0.1 -> x1=0.45*3840=1728, x2=0.55*3840=2112
        # y_center=0.5, height=0.1 -> y1=0.45*2160=972, y2=0.55*2160=1188
        assert result == (1728.0, 972.0, 2112.0, 1188.0)

    def test_asymmetric_box(self):
        """Test conversion with asymmetric box (different width/height)."""
        xywh = (0.3, 0.7, 0.6, 0.2)
        image_size = (500, 400)

        result = xywh_to_xyxy(xywh, image_size)

        # x_center=0.3, width=0.6 -> x1=0.0*500=0, x2=0.6*500=300
        # y_center=0.7, height=0.2 -> y1=0.6*400=240, y2=0.8*400=320
        assert result == (0.0, 240.0, 300.0, 320.0)

    def test_floating_point_precision(self):
        """Test that floating point precision is maintained."""
        xywh = (0.333333, 0.666666, 0.123456, 0.789012)
        image_size = (1000, 1000)

        result = xywh_to_xyxy(xywh, image_size)

        # Verify calculations are precise
        x1 = (0.333333 - 0.123456 / 2) * 1000
        y1 = (0.666666 - 0.789012 / 2) * 1000
        x2 = (0.333333 + 0.123456 / 2) * 1000
        y2 = (0.666666 + 0.789012 / 2) * 1000

        assert abs(result[0] - x1) < 1e-6
        assert abs(result[1] - y1) < 1e-6
        assert abs(result[2] - x2) < 1e-6
        assert abs(result[3] - y2) < 1e-6


class TestXywhsToXyxys:
    """Test cases for xywhs_to_xyxys batch conversion."""

    def test_single_box(self):
        """Test conversion of single box in list."""
        xywhs = [(0.5, 0.5, 0.4, 0.6)]
        image_size = (100, 100)

        result = xywhs_to_xyxys(xywhs, image_size)

        assert len(result) == 1
        assert result[0] == (30.0, 20.0, 70.0, 80.0)

    def test_multiple_boxes(self):
        """Test conversion of multiple boxes."""
        xywhs = [
            (0.5, 0.5, 0.4, 0.6),
            (0.2, 0.3, 0.2, 0.2),
            (0.8, 0.7, 0.3, 0.4),
        ]
        image_size = (100, 100)

        result = xywhs_to_xyxys(xywhs, image_size)

        assert len(result) == 3
        assert result[0] == pytest.approx((30.0, 20.0, 70.0, 80.0))
        assert result[1] == pytest.approx((10.0, 20.0, 30.0, 40.0))
        assert result[2] == pytest.approx((65.0, 50.0, 95.0, 90.0))

    def test_empty_list(self):
        """Test conversion of empty list."""
        xywhs = []
        image_size = (100, 100)

        result = xywhs_to_xyxys(xywhs, image_size)

        assert result == []

    def test_many_boxes(self):
        """Test conversion of many boxes."""
        xywhs = [(0.5, 0.5, 0.1, 0.1) for _ in range(100)]
        image_size = (100, 100)

        result = xywhs_to_xyxys(xywhs, image_size)

        assert len(result) == 100
        # All boxes should be the same
        for box in result:
            assert box == pytest.approx((45.0, 45.0, 55.0, 55.0))


class TestXyxyToXywh:
    """Test cases for xyxy_to_xywh conversion."""

    def test_centered_box(self):
        """Test conversion of a centered box."""
        xyxy = (30.0, 20.0, 70.0, 80.0)
        image_size = (100, 100)

        result = xyxy_to_xywh(xyxy, image_size)

        # x1=30, x2=70 -> x_center=(30+70)/2/100=0.5, width=(70-30)/100=0.4
        # y1=20, y2=80 -> y_center=(20+80)/2/100=0.5, height=(80-20)/100=0.6
        assert result == (0.5, 0.5, 0.4, 0.6)

    def test_top_left_corner(self):
        """Test conversion of box at top-left corner."""
        xyxy = (0.0, 0.0, 20.0, 20.0)
        image_size = (100, 100)

        result = xyxy_to_xywh(xyxy, image_size)

        assert result == (0.1, 0.1, 0.2, 0.2)

    def test_bottom_right_corner(self):
        """Test conversion of box at bottom-right corner."""
        xyxy = (80.0, 80.0, 100.0, 100.0)
        image_size = (100, 100)

        result = xyxy_to_xywh(xyxy, image_size)

        assert result == (0.9, 0.9, 0.2, 0.2)

    def test_rectangular_image_wide(self):
        """Test conversion with wide rectangular image."""
        xyxy = (60.0, 40.0, 140.0, 60.0)
        image_size = (200, 100)

        result = xyxy_to_xywh(xyxy, image_size)

        # x1=60, x2=140 -> x_center=(60+140)/2/200=0.5, width=(140-60)/200=0.4
        # y1=40, y2=60 -> y_center=(40+60)/2/100=0.5, height=(60-40)/100=0.2
        assert result == (0.5, 0.5, 0.4, 0.2)

    def test_rectangular_image_tall(self):
        """Test conversion with tall rectangular image."""
        xyxy = (40.0, 60.0, 60.0, 140.0)
        image_size = (100, 200)

        result = xyxy_to_xywh(xyxy, image_size)

        # x1=40, x2=60 -> x_center=(40+60)/2/100=0.5, width=(60-40)/100=0.2
        # y1=60, y2=140 -> y_center=(60+140)/2/200=0.5, height=(140-60)/200=0.4
        assert result == (0.5, 0.5, 0.2, 0.4)

    def test_very_small_box(self):
        """Test conversion of very small box."""
        xyxy = (495.0, 495.0, 505.0, 505.0)
        image_size = (1000, 1000)

        result = xyxy_to_xywh(xyxy, image_size)

        assert result == (0.5, 0.5, 0.01, 0.01)

    def test_very_large_box(self):
        """Test conversion of very large box (entire image)."""
        xyxy = (0.0, 0.0, 100.0, 100.0)
        image_size = (100, 100)

        result = xyxy_to_xywh(xyxy, image_size)

        assert result == (0.5, 0.5, 1.0, 1.0)

    def test_zero_area_box(self):
        """Test conversion with zero area box (edge case)."""
        xyxy = (50.0, 50.0, 50.0, 50.0)
        image_size = (100, 100)

        result = xyxy_to_xywh(xyxy, image_size)

        assert result == (0.5, 0.5, 0.0, 0.0)

    def test_large_image_dimensions(self):
        """Test conversion with large image dimensions."""
        xyxy = (768.0, 378.0, 1152.0, 702.0)
        image_size = (1920, 1080)

        result = xyxy_to_xywh(xyxy, image_size)

        # x1=768, x2=1152 -> x_center=(768+1152)/2/1920=0.5, width=(1152-768)/1920=0.2
        # y1=378, y2=702 -> y_center=(378+702)/2/1080=0.5, height=(702-378)/1080=0.3
        assert result == (0.5, 0.5, 0.2, 0.3)

    def test_4k_resolution(self):
        """Test conversion with 4K resolution."""
        xyxy = (1728.0, 972.0, 2112.0, 1188.0)
        image_size = (3840, 2160)

        result = xyxy_to_xywh(xyxy, image_size)

        assert result == (0.5, 0.5, 0.1, 0.1)

    def test_asymmetric_box(self):
        """Test conversion with asymmetric box."""
        xyxy = (0.0, 240.0, 300.0, 320.0)
        image_size = (500, 400)

        result = xyxy_to_xywh(xyxy, image_size)

        # x1=0, x2=300 -> x_center=(0+300)/2/500=0.3, width=(300-0)/500=0.6
        # y1=240, y2=320 -> y_center=(240+320)/2/400=0.7, height=(320-240)/400=0.2
        assert result == (0.3, 0.7, 0.6, 0.2)

    def test_floating_point_precision(self):
        """Test that floating point precision is maintained."""
        xyxy = (271.605, 272.160, 395.061, 666.672)
        image_size = (1000, 1000)

        result = xyxy_to_xywh(xyxy, image_size)

        # Verify calculations
        x_center = (271.605 + 395.061) / 2 / 1000
        y_center = (272.160 + 666.672) / 2 / 1000
        width = (395.061 - 271.605) / 1000
        height = (666.672 - 272.160) / 1000

        assert abs(result[0] - x_center) < 1e-6
        assert abs(result[1] - y_center) < 1e-6
        assert abs(result[2] - width) < 1e-6
        assert abs(result[3] - height) < 1e-6


class TestXyxysToXywhs:
    """Test cases for xyxys_to_xywhs batch conversion."""

    def test_single_box(self):
        """Test conversion of single box in list."""
        xyxys = [(30.0, 20.0, 70.0, 80.0)]
        image_size = (100, 100)

        result = xyxys_to_xywhs(xyxys, image_size)

        assert len(result) == 1
        assert result[0] == (0.5, 0.5, 0.4, 0.6)

    def test_multiple_boxes(self):
        """Test conversion of multiple boxes."""
        xyxys = [
            (30.0, 20.0, 70.0, 80.0),
            (10.0, 20.0, 30.0, 40.0),
            (65.0, 50.0, 95.0, 90.0),
        ]
        image_size = (100, 100)

        result = xyxys_to_xywhs(xyxys, image_size)

        assert len(result) == 3
        assert result[0] == (0.5, 0.5, 0.4, 0.6)
        assert result[1] == (0.2, 0.3, 0.2, 0.2)
        assert result[2] == (0.8, 0.7, 0.3, 0.4)

    def test_empty_list(self):
        """Test conversion of empty list."""
        xyxys = []
        image_size = (100, 100)

        result = xyxys_to_xywhs(xyxys, image_size)

        assert result == []

    def test_numpy_array_input(self):
        """Test conversion with numpy array input."""
        xyxys = np.array([
            [30.0, 20.0, 70.0, 80.0],
            [10.0, 20.0, 30.0, 40.0],
        ])
        image_size = (100, 100)

        result = xyxys_to_xywhs(xyxys, image_size)

        assert len(result) == 2
        assert result[0] == (0.5, 0.5, 0.4, 0.6)
        assert result[1] == (0.2, 0.3, 0.2, 0.2)

    def test_many_boxes(self):
        """Test conversion of many boxes."""
        xyxys = [(45.0, 45.0, 55.0, 55.0) for _ in range(100)]
        image_size = (100, 100)

        result = xyxys_to_xywhs(xyxys, image_size)

        assert len(result) == 100
        # All boxes should be the same
        for box in result:
            assert box == (0.5, 0.5, 0.1, 0.1)


class TestRoundTripConversions:
    """Test that conversions are reversible (round-trip tests)."""

    def test_xywh_to_xyxy_to_xywh(self):
        """Test round-trip conversion xywh -> xyxy -> xywh."""
        original = (0.5, 0.5, 0.4, 0.6)
        image_size = (100, 100)

        xyxy = xywh_to_xyxy(original, image_size)
        result = xyxy_to_xywh(xyxy, image_size)

        assert result == original

    def test_xyxy_to_xywh_to_xyxy(self):
        """Test round-trip conversion xyxy -> xywh -> xyxy."""
        original = (30.0, 20.0, 70.0, 80.0)
        image_size = (100, 100)

        xywh = xyxy_to_xywh(original, image_size)
        result = xywh_to_xyxy(xywh, image_size)

        assert result == original

    def test_batch_round_trip_xywh(self):
        """Test batch round-trip conversion for xywh."""
        original = [
            (0.5, 0.5, 0.4, 0.6),
            (0.2, 0.3, 0.2, 0.2),
            (0.8, 0.7, 0.3, 0.4),
        ]
        image_size = (100, 100)

        xyxys = xywhs_to_xyxys(original, image_size)
        result = xyxys_to_xywhs(xyxys, image_size)

        for orig, res in zip(original, result):
            assert res == pytest.approx(orig)

    def test_batch_round_trip_xyxy(self):
        """Test batch round-trip conversion for xyxy."""
        original = [
            (30.0, 20.0, 70.0, 80.0),
            (10.0, 20.0, 30.0, 40.0),
            (65.0, 50.0, 95.0, 90.0),
        ]
        image_size = (100, 100)

        xywhs = xyxys_to_xywhs(original, image_size)
        result = xywhs_to_xyxys(xywhs, image_size)

        for orig, res in zip(original, result):
            assert res == pytest.approx(orig)

    def test_round_trip_with_floating_point(self):
        """Test round-trip preserves floating point precision."""
        original = (0.333333, 0.666666, 0.123456, 0.789012)
        image_size = (1000, 1000)

        xyxy = xywh_to_xyxy(original, image_size)
        result = xyxy_to_xywh(xyxy, image_size)

        # Should be exact for normalized coordinates
        for orig, res in zip(original, result):
            assert abs(orig - res) < 1e-10

    def test_round_trip_edge_cases(self):
        """Test round-trip with edge case values."""
        edge_cases = [
            (0.0, 0.0, 0.0, 0.0),  # Zero box
            (0.5, 0.5, 1.0, 1.0),  # Full image
            (0.1, 0.1, 0.2, 0.2),  # Top-left
            (0.9, 0.9, 0.2, 0.2),  # Bottom-right
        ]
        image_size = (100, 100)

        for original in edge_cases:
            xyxy = xywh_to_xyxy(original, image_size)
            result = xyxy_to_xywh(xyxy, image_size)
            assert result == original
