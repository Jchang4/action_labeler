"""Test cases for the yolov8_labels_to_rows helper function."""

import tempfile
from pathlib import Path

import pytest

from action_labeler.detections.helpers import yolov8_labels_to_rows


class TestYolov8LabelsToRows:
    """Test cases for the yolov8_labels_to_rows helper function."""

    def test_single_bbox_detection(self, tmp_path):
        """Test parsing a single bounding box detection."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_multiple_bbox_detections(self, tmp_path):
        """Test parsing multiple bounding box detections."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.15 0.25\n2 0.8 0.2 0.1 0.1\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 3
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]
        assert result[1] == [1.0, 0.3, 0.7, 0.15, 0.25]
        assert result[2] == [2.0, 0.8, 0.2, 0.1, 0.1]

    def test_segmentation_detection(self, tmp_path):
        """Test parsing a segmentation detection with polygon points."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]

    def test_pose_detection(self, tmp_path):
        """Test parsing a pose detection with keypoints."""
        label_file = tmp_path / "test.txt"
        # 5 bbox values + 4 keypoint coords (2 keypoints)
        label_file.write_text("0 0.5 0.5 0.2 0.3 0.6 0.3 0.5 0.4\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.3, 0.5, 0.4]

    def test_empty_file(self, tmp_path):
        """Test parsing an empty file returns empty list."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("")

        result = yolov8_labels_to_rows(label_file)

        assert result == []

    def test_whitespace_only_file(self, tmp_path):
        """Test parsing a file with only whitespace returns empty list."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("   \n\n  \n\t\n")

        result = yolov8_labels_to_rows(label_file)

        assert result == []

    def test_empty_lines_between_detections(self, tmp_path):
        """Test that empty lines between detections are handled correctly."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n\n1 0.3 0.7 0.15 0.25\n\n\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 2
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]
        assert result[1] == [1.0, 0.3, 0.7, 0.15, 0.25]

    def test_leading_and_trailing_whitespace(self, tmp_path):
        """Test that leading/trailing whitespace is properly stripped."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("  0 0.5 0.5 0.2 0.3  \n\t1 0.3 0.7 0.15 0.25\t\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 2
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]
        assert result[1] == [1.0, 0.3, 0.7, 0.15, 0.25]

    def test_single_value_line_ignored(self, tmp_path):
        """Test that lines with only one value (class_id) are ignored."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0\n0 0.5 0.5 0.2 0.3\n")

        result = yolov8_labels_to_rows(label_file)

        # Single value line should be ignored
        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_accepts_path_string(self, tmp_path):
        """Test that function accepts path as string."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n")

        # Pass as string instead of Path
        result = yolov8_labels_to_rows(str(label_file))

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_accepts_path_object(self, tmp_path):
        """Test that function accepts Path object."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n")

        # Pass as Path object
        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_integer_class_ids(self, tmp_path):
        """Test parsing with integer class IDs."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n5 0.3 0.7 0.15 0.25\n")

        result = yolov8_labels_to_rows(label_file)

        assert result[0][0] == 0.0
        assert result[1][0] == 5.0

    def test_float_class_ids(self, tmp_path):
        """Test parsing with float class IDs (edge case)."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0.0 0.5 0.5 0.2 0.3\n1.5 0.3 0.7 0.15 0.25\n")

        result = yolov8_labels_to_rows(label_file)

        assert result[0][0] == 0.0
        assert result[1][0] == 1.5

    def test_large_number_of_detections(self, tmp_path):
        """Test parsing a file with many detections."""
        label_file = tmp_path / "test.txt"
        lines = [f"{i % 10} 0.5 0.5 0.2 0.3\n" for i in range(100)]
        label_file.write_text("".join(lines))

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 100
        for i, row in enumerate(result):
            assert row[0] == float(i % 10)

    def test_very_small_values(self, tmp_path):
        """Test parsing with very small normalized values."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.001 0.002 0.003 0.004\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.001, 0.002, 0.003, 0.004]

    def test_values_near_one(self, tmp_path):
        """Test parsing with values near 1.0."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.999 0.998 0.997 0.996\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.999, 0.998, 0.997, 0.996]

    def test_exact_boundary_values(self, tmp_path):
        """Test parsing with exact 0.0 and 1.0 values."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.0 1.0 1.0 0.0\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.0, 1.0, 1.0, 0.0]

    def test_scientific_notation(self, tmp_path):
        """Test parsing with scientific notation."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 1e-3 2e-3 3e-3 4e-3\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.001, 0.002, 0.003, 0.004]

    def test_negative_values(self, tmp_path):
        """Test parsing with negative values (edge case - invalid but parseable)."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 -0.1 -0.2 0.3 0.4\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, -0.1, -0.2, 0.3, 0.4]

    def test_values_greater_than_one(self, tmp_path):
        """Test parsing with values >1.0 (edge case - invalid but parseable)."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 1.5 2.0 0.3 0.4\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 1.5, 2.0, 0.3, 0.4]

    def test_utf8_encoding(self, tmp_path):
        """Test that UTF-8 encoding is handled correctly."""
        label_file = tmp_path / "test.txt"
        # Write valid YOLO data with UTF-8 encoding
        label_file.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_mixed_line_endings_unix(self, tmp_path):
        """Test parsing with Unix line endings."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.15 0.25\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 2

    def test_mixed_line_endings_windows(self, tmp_path):
        """Test parsing with Windows line endings."""
        label_file = tmp_path / "test.txt"
        label_file.write_bytes(b"0 0.5 0.5 0.2 0.3\r\n1 0.3 0.7 0.15 0.25\r\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 2

    def test_no_trailing_newline(self, tmp_path):
        """Test parsing file without trailing newline."""
        label_file = tmp_path / "test.txt"
        # No \n at the end
        label_file.write_text("0 0.5 0.5 0.2 0.3")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_multiple_spaces_between_values(self, tmp_path):
        """Test parsing with multiple spaces between values."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0    0.5    0.5    0.2    0.3\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_tabs_between_values(self, tmp_path):
        """Test parsing with tabs between values."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0\t0.5\t0.5\t0.2\t0.3\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_mixed_spaces_and_tabs(self, tmp_path):
        """Test parsing with mixed spaces and tabs."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0  \t 0.5\t  0.5 \t0.2   0.3\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert result[0] == [0.0, 0.5, 0.5, 0.2, 0.3]

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            yolov8_labels_to_rows("/nonexistent/path/file.txt")

    def test_invalid_number_format(self, tmp_path):
        """Test that ValueError is raised for invalid number format."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 abc 0.5 0.2 0.3\n")

        with pytest.raises(ValueError):
            yolov8_labels_to_rows(label_file)

    def test_large_polygon_segmentation(self, tmp_path):
        """Test parsing a segmentation with many polygon points."""
        label_file = tmp_path / "test.txt"
        # Create a polygon with 20 points (40 coordinates)
        coords = [str(i * 0.05) for i in range(40)]
        label_file.write_text(f"0 {' '.join(coords)}\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert len(result[0]) == 41  # class_id + 40 coordinates

    def test_coco_pose_17_keypoints(self, tmp_path):
        """Test parsing COCO pose format with 17 keypoints."""
        label_file = tmp_path / "test.txt"
        # 5 bbox values + 34 keypoint coords (17 keypoints * 2)
        bbox = "0 0.5 0.5 0.2 0.3"
        keypoints = " ".join([str(i * 0.01) for i in range(34)])
        label_file.write_text(f"{bbox} {keypoints}\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        assert len(result[0]) == 39  # 5 + 34

    def test_preserves_float_precision(self, tmp_path):
        """Test that float precision is preserved during parsing."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.123456789 0.987654321 0.246802468 0.135791357\n")

        result = yolov8_labels_to_rows(label_file)

        assert len(result) == 1
        # Check that precision is maintained (within floating point limits)
        assert abs(result[0][1] - 0.123456789) < 1e-9
        assert abs(result[0][2] - 0.987654321) < 1e-9
