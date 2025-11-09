"""Test cases for the detect_format helper function."""

import pytest

from action_labeler.detections.helpers import DetectionFormat, detect_format


class TestDetectFormat:
    """Test cases for the detect_format helper function."""

    def test_empty_rows_returns_bbox(self):
        """Test that empty rows default to BBOX format."""
        rows = []
        assert detect_format(rows) == DetectionFormat.BBOX

    def test_bbox_format_5_values(self):
        """Test detection of BBOX format with exactly 5 values."""
        rows = [[0, 0.5, 0.5, 0.2, 0.3]]
        assert detect_format(rows) == DetectionFormat.BBOX

    def test_bbox_format_multiple_rows(self):
        """Test detection of BBOX format with multiple rows."""
        rows = [
            [0, 0.5, 0.5, 0.2, 0.3],
            [1, 0.2, 0.2, 0.1, 0.1],
            [2, 0.8, 0.8, 0.2, 0.1],
        ]
        assert detect_format(rows) == DetectionFormat.BBOX

    def test_segment_format_simple(self):
        """Test detection of SEGMENT format with polygon points."""
        # 9 values: class_id + 8 coords (4 points for a rectangle)
        rows = [[0, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]]
        assert detect_format(rows) == DetectionFormat.SEGMENT

    def test_segment_format_odd_number_points(self):
        """Test detection of SEGMENT format with odd number of polygon points."""
        # 11 values: class_id + 10 coords (5 points)
        rows = [[0, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4, 0.2, 0.3]]
        assert detect_format(rows) == DetectionFormat.SEGMENT

    def test_segment_format_even_number_points(self):
        """Test detection of SEGMENT format with even number of polygon points."""
        # 13 values: class_id + 12 coords (6 points)
        rows = [[0, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4, 0.2, 0.3, 0.15, 0.25]]
        assert detect_format(rows) == DetectionFormat.SEGMENT

    def test_pose_format_2_keypoints(self):
        """Test detection of POSE format with 2 keypoints."""
        # 9 values: 5 bbox + 4 keypoint coords (2 keypoints * 2)
        rows = [[0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.3, 0.5, 0.4]]
        assert detect_format(rows, num_keypoints=2) == DetectionFormat.POSE

    def test_pose_format_3_keypoints(self):
        """Test detection of POSE format with 3 keypoints."""
        # 11 values: 5 bbox + 6 keypoint coords (3 keypoints * 2)
        rows = [[0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.3, 0.5, 0.4, 0.4, 0.3]]
        assert detect_format(rows, num_keypoints=3) == DetectionFormat.POSE

    def test_pose_format_17_keypoints_coco(self):
        """Test detection of POSE format with 17 keypoints (COCO format)."""
        # 39 values: 5 bbox + 34 keypoint coords (17 keypoints * 2)
        row = [0, 0.5, 0.5, 0.2, 0.3] + [0.1 * i for i in range(34)]
        rows = [row]
        assert detect_format(rows, num_keypoints=17) == DetectionFormat.POSE

    def test_pose_format_multiple_detections(self):
        """Test POSE format detection with multiple people in one image."""
        # Multiple rows with same format
        rows = [
            [0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.3, 0.5, 0.4],  # person 1
            [0, 0.3, 0.7, 0.15, 0.25, 0.35, 0.65, 0.25, 0.75],  # person 2
        ]
        assert detect_format(rows, num_keypoints=2) == DetectionFormat.POSE

    def test_segment_without_num_keypoints(self):
        """Test that even-valued rows default to SEGMENT without num_keypoints."""
        # 9 values could be pose with 2 keypoints, but without num_keypoints it's segment
        rows = [[0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.3, 0.5, 0.4]]
        assert detect_format(rows) == DetectionFormat.SEGMENT

    def test_segment_wrong_keypoint_count(self):
        """Test that wrong keypoint count falls back to SEGMENT."""
        # 11 values but we expect 17 keypoints (which would be 5 + 34 = 39 values)
        rows = [[0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.3, 0.5, 0.4, 0.4, 0.3]]
        assert detect_format(rows, num_keypoints=17) == DetectionFormat.SEGMENT

    def test_segment_when_pose_values_dont_match(self):
        """Test SEGMENT is returned when values don't match expected pose format."""
        # 13 values with num_keypoints=2 (expected: 5 + 4 = 9)
        rows = [[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
        assert detect_format(rows, num_keypoints=2) == DetectionFormat.SEGMENT

    def test_invalid_too_few_values(self):
        """Test that fewer than 5 values raises ValueError."""
        rows = [[0, 0.5, 0.5]]  # Only 3 values
        with pytest.raises(ValueError, match="expected at least 5 values"):
            detect_format(rows)

    def test_invalid_4_values(self):
        """Test that exactly 4 values raises ValueError."""
        rows = [[0, 0.5, 0.5, 0.2]]
        with pytest.raises(ValueError, match="expected at least 5 values, got 4"):
            detect_format(rows)

    def test_invalid_1_value(self):
        """Test that only class_id raises ValueError."""
        rows = [[0]]
        with pytest.raises(ValueError, match="expected at least 5 values, got 1"):
            detect_format(rows)

    def test_pose_with_zero_keypoints(self):
        """Test edge case: num_keypoints=0 returns BBOX (since 5 values matches BBOX first)."""
        # 5 values with num_keypoints=0 (5 + 2*0 = 5)
        rows = [[0, 0.5, 0.5, 0.2, 0.3]]
        # BBOX check happens before POSE check, so this returns BBOX
        assert detect_format(rows, num_keypoints=0) == DetectionFormat.BBOX

    def test_large_segmentation_polygon(self):
        """Test SEGMENT format with many polygon points."""
        # 51 values: class_id + 50 coords (25 points)
        row = [0] + [0.1 * i for i in range(50)]
        rows = [row]
        assert detect_format(rows) == DetectionFormat.SEGMENT

    def test_first_row_determines_format(self):
        """Test that only the first row is checked for format detection."""
        # First row is bbox format (5 values)
        # Even if subsequent rows would have different lengths (they shouldn't in practice)
        rows = [[0, 0.5, 0.5, 0.2, 0.3]]
        assert detect_format(rows) == DetectionFormat.BBOX

    def test_num_keypoints_none_with_segment_data(self):
        """Test that None num_keypoints with >5 values returns SEGMENT."""
        rows = [[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
        assert detect_format(rows, num_keypoints=None) == DetectionFormat.SEGMENT

    def test_float_values_in_rows(self):
        """Test that float values are handled correctly."""
        rows = [[0.0, 0.5, 0.5, 0.2, 0.3]]
        assert detect_format(rows) == DetectionFormat.BBOX

    def test_mixed_int_float_values(self):
        """Test that mixed int/float values work correctly."""
        rows = [[0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.7]]  # 7 values - segment
        assert detect_format(rows) == DetectionFormat.SEGMENT
