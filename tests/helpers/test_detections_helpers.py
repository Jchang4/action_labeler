from pathlib import Path

import pytest

from action_labeler.helpers.detections_helpers import (
    image_to_txt_path,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


class TestImageToTxtPath:
    """Test cases for the image_to_txt_path function."""

    def test_basic_functionality_default_detect(self):
        """Test basic functionality with default detection_type='detect'."""
        image_path = Path("/path/to/dataset/images/photo.jpg")
        expected = Path("/path/to/dataset/detect/photo.txt")
        result = image_to_txt_path(image_path)
        assert result == expected

    def test_custom_detection_type_segment(self):
        """Test with custom detection_type='segment'."""
        image_path = Path("/path/to/dataset/images/photo.jpg")
        expected = Path("/path/to/dataset/segment/photo.txt")
        result = image_to_txt_path(image_path, detection_type="segment")
        assert result == expected

    @pytest.mark.parametrize(
        "extension", [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]
    )
    def test_different_image_extensions(self, extension):
        """Test that different image extensions all convert to .txt."""
        image_path = Path(f"/path/to/dataset/images/photo{extension}")
        expected = Path("/path/to/dataset/detect/photo.txt")
        result = image_to_txt_path(image_path)
        assert result == expected

    def test_path_vs_string_input(self):
        """Test that both Path objects and string paths work identically."""
        path_str = "/path/to/dataset/images/photo.jpg"
        path_obj = Path(path_str)

        result_from_str = image_to_txt_path(path_str)
        result_from_path = image_to_txt_path(path_obj)

        assert result_from_str == result_from_path
        assert result_from_str == Path("/path/to/dataset/detect/photo.txt")

    def test_nested_directory_structure(self):
        """Test with deeper directory structures."""
        image_path = Path("/deep/nested/path/to/dataset/images/photo.jpg")
        expected = Path("/deep/nested/path/to/dataset/detect/photo.txt")
        result = image_to_txt_path(image_path)
        assert result == expected

    def test_filename_with_multiple_dots(self):
        """Test filenames with multiple dots in the name."""
        image_path = Path("/path/to/dataset/images/image.backup.jpg")
        expected = Path("/path/to/dataset/detect/image.backup.txt")
        result = image_to_txt_path(image_path)
        assert result == expected

    def test_filename_with_spaces(self):
        """Test filenames with spaces."""
        image_path = Path("/path/to/dataset/images/my photo.jpg")
        expected = Path("/path/to/dataset/detect/my photo.txt")
        result = image_to_txt_path(image_path)
        assert result == expected

    @pytest.mark.parametrize(
        "detection_type", ["detect", "segment", "classify", "custom_type"]
    )
    def test_various_detection_types(self, detection_type):
        """Test various detection types."""
        image_path = Path("/path/to/dataset/images/photo.jpg")
        expected = Path(f"/path/to/dataset/{detection_type}/photo.txt")
        result = image_to_txt_path(image_path, detection_type=detection_type)
        assert result == expected

    def test_relative_path(self):
        """Test with relative paths."""
        image_path = Path("dataset/images/photo.jpg")
        expected = Path("dataset/detect/photo.txt")
        result = image_to_txt_path(image_path)
        assert result == expected

    def test_single_directory_level(self):
        """Test with minimal directory structure."""
        image_path = Path("images/photo.jpg")
        expected = Path("detect/photo.txt")
        result = image_to_txt_path(image_path)
        assert result == expected

    def test_return_type_is_path(self):
        """Test that the function returns a Path object."""
        image_path = "/path/to/dataset/images/photo.jpg"
        result = image_to_txt_path(image_path)
        assert isinstance(result, Path)


class TestXyxyToXywh:
    """Test cases for the xyxy_to_xywh function."""

    def test_basic_conversion(self):
        """Test basic xyxy to xywh conversion."""
        # Box from (100, 50) to (200, 150) in a 640x480 image
        xyxy = (100, 50, 200, 150)
        image_size = (640, 480)

        result = xyxy_to_xywh(xyxy, image_size)

        # Expected: center_x=150/640=0.234375, center_y=100/480=0.208333,
        #           width=100/640=0.15625, height=100/480=0.208333
        expected = [0.234375, 0.20833333333333334, 0.15625, 0.20833333333333334]
        assert result == expected

    def test_full_image_box(self):
        """Test conversion of a box that covers the entire image."""
        xyxy = (0, 0, 640, 480)
        image_size = (640, 480)

        result = xyxy_to_xywh(xyxy, image_size)

        # Expected: center at (0.5, 0.5), full width and height (1.0, 1.0)
        expected = [0.5, 0.5, 1.0, 1.0]
        assert result == expected

    def test_small_box_in_corner(self):
        """Test conversion of a small box in the top-left corner."""
        xyxy = (0, 0, 10, 10)
        image_size = (640, 480)

        result = xyxy_to_xywh(xyxy, image_size)

        # Expected: center_x=5/640, center_y=5/480, width=10/640, height=10/480
        expected = [5 / 640, 5 / 480, 10 / 640, 10 / 480]
        assert result == expected

    def test_center_box(self):
        """Test conversion of a box in the center of the image."""
        xyxy = (270, 190, 370, 290)  # 100x100 box centered in 640x480 image
        image_size = (640, 480)

        result = xyxy_to_xywh(xyxy, image_size)

        # Expected: center_x=320/640=0.5, center_y=240/480=0.5,
        #           width=100/640=0.15625, height=100/480=0.208333
        expected = [0.5, 0.5, 0.15625, 0.20833333333333334]
        assert result == expected

    @pytest.mark.parametrize(
        "xyxy,image_size,expected",
        [
            # Square box in square image
            ((0, 0, 100, 100), (200, 200), [0.25, 0.25, 0.5, 0.5]),
            # Rectangular box in rectangular image
            ((50, 25, 150, 75), (400, 300), [0.25, 1 / 6, 0.25, 1 / 6]),
            # Single pixel box
            (
                (100, 100, 101, 101),
                (640, 480),
                [100.5 / 640, 100.5 / 480, 1 / 640, 1 / 480],
            ),
        ],
    )
    def test_various_boxes_and_image_sizes(self, xyxy, image_size, expected):
        """Test various combinations of boxes and image sizes."""
        result = xyxy_to_xywh(xyxy, image_size)

        # Use approximate equality for floating point comparisons
        for i in range(4):
            assert abs(result[i] - expected[i]) < 1e-10

    def test_float_coordinates(self):
        """Test with floating point coordinates."""
        xyxy = (100.5, 50.7, 200.3, 150.9)
        image_size = (640, 480)

        result = xyxy_to_xywh(xyxy, image_size)

        # Calculate expected values
        center_x = (100.5 + 200.3) / 2 / 640
        center_y = (50.7 + 150.9) / 2 / 480
        width = (200.3 - 100.5) / 640
        height = (150.9 - 50.7) / 480

        expected = [center_x, center_y, width, height]
        assert result == expected

    def test_return_type_is_list(self):
        """Test that the function returns a list (despite type hint saying tuple)."""
        xyxy = (100, 50, 200, 150)
        image_size = (640, 480)

        result = xyxy_to_xywh(xyxy, image_size)

        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(coord, float) for coord in result)

    def test_zero_width_box(self):
        """Test edge case with zero width box."""
        xyxy = (100, 50, 100, 150)  # Zero width
        image_size = (640, 480)

        result = xyxy_to_xywh(xyxy, image_size)

        expected = [100 / 640, 100 / 480, 0.0, 100 / 480]
        assert result == expected

    def test_zero_height_box(self):
        """Test edge case with zero height box."""
        xyxy = (100, 50, 200, 50)  # Zero height
        image_size = (640, 480)

        result = xyxy_to_xywh(xyxy, image_size)

        expected = [150 / 640, 50 / 480, 100 / 640, 0.0]
        assert result == expected

    def test_different_input_types(self):
        """Test that function works with different input types (list, tuple)."""
        image_size = (640, 480)

        # Test with tuple
        xyxy_tuple = (100, 50, 200, 150)
        result_tuple = xyxy_to_xywh(xyxy_tuple, image_size)

        # Test with list
        xyxy_list = [100, 50, 200, 150]
        result_list = xyxy_to_xywh(xyxy_list, image_size)

        assert result_tuple == result_list


class TestXywhToXyxy:
    """Test cases for the xywh_to_xyxy function."""

    def test_basic_conversion(self):
        """Test basic xywh to xyxy conversion."""
        # Box with center at (0.234375, 0.208333), width=0.15625, height=0.208333 in a 640x480 image
        xywh = (0.234375, 0.20833333333333334, 0.15625, 0.20833333333333334)
        image_size = (640, 480)

        result = xywh_to_xyxy(xywh, image_size)

        # Expected: (100, 50, 200, 150)
        expected = (100.0, 50.0, 200.0, 150.0)
        assert result == expected

    def test_full_image_box(self):
        """Test conversion of a box that covers the entire image."""
        xywh = (0.5, 0.5, 1.0, 1.0)
        image_size = (640, 480)

        result = xywh_to_xyxy(xywh, image_size)

        # Expected: (0, 0, 640, 480)
        expected = (0.0, 0.0, 640.0, 480.0)
        assert result == expected

    def test_small_box_in_corner(self):
        """Test conversion of a small box in the top-left corner."""
        xywh = (5 / 640, 5 / 480, 10 / 640, 10 / 480)
        image_size = (640, 480)

        result = xywh_to_xyxy(xywh, image_size)

        # Expected: (0, 0, 10, 10)
        expected = (0.0, 0.0, 10.0, 10.0)
        assert result == expected

    def test_center_box(self):
        """Test conversion of a box in the center of the image."""
        xywh = (0.5, 0.5, 0.15625, 0.20833333333333334)
        image_size = (640, 480)

        result = xywh_to_xyxy(xywh, image_size)

        # Expected: box centered in 640x480 image
        expected = (270.0, 190.0, 370.0, 290.0)
        assert result == expected

    @pytest.mark.parametrize(
        "xywh,image_size,expected",
        [
            # Square box in square image
            ((0.25, 0.25, 0.5, 0.5), (200, 200), (0.0, 0.0, 100.0, 100.0)),
            # Rectangular box in rectangular image
            ((0.25, 1 / 6, 0.25, 1 / 6), (400, 300), (50.0, 25.0, 150.0, 75.0)),
            # Single pixel box
            (
                (100.5 / 640, 100.5 / 480, 1 / 640, 1 / 480),
                (640, 480),
                (100.0, 100.0, 101.0, 101.0),
            ),
        ],
    )
    def test_various_boxes_and_image_sizes(self, xywh, image_size, expected):
        """Test various combinations of boxes and image sizes."""
        result = xywh_to_xyxy(xywh, image_size)

        # Use approximate equality for floating point comparisons
        for i in range(4):
            assert abs(result[i] - expected[i]) < 1e-10

    def test_float_coordinates(self):
        """Test with floating point coordinates."""
        xywh = (0.2345, 0.2098, 0.1560, 0.2085)
        image_size = (640, 480)

        result = xywh_to_xyxy(xywh, image_size)

        # Calculate expected values
        x_center = xywh[0] * 640
        y_center = xywh[1] * 480
        width = xywh[2] * 640
        height = xywh[3] * 480

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        expected = (x1, y1, x2, y2)
        assert result == expected

    def test_return_type_is_tuple(self):
        """Test that the function returns a tuple."""
        xywh = (0.234375, 0.20833333333333334, 0.15625, 0.20833333333333334)
        image_size = (640, 480)

        result = xywh_to_xyxy(xywh, image_size)

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(coord, float) for coord in result)

    def test_zero_width_box(self):
        """Test edge case with zero width box."""
        xywh = (0.5, 0.5, 0.0, 0.2)  # Zero width
        image_size = (640, 480)

        result = xywh_to_xyxy(xywh, image_size)

        expected = (320.0, 192.0, 320.0, 288.0)
        assert result == expected

    def test_zero_height_box(self):
        """Test edge case with zero height box."""
        xywh = (0.5, 0.5, 0.2, 0.0)  # Zero height
        image_size = (640, 480)

        result = xywh_to_xyxy(xywh, image_size)

        expected = (256.0, 240.0, 384.0, 240.0)
        assert result == expected

    def test_different_input_types(self):
        """Test that function works with different input types (list, tuple)."""
        image_size = (640, 480)

        # Test with tuple
        xywh_tuple = (0.5, 0.5, 0.2, 0.2)
        result_tuple = xywh_to_xyxy(xywh_tuple, image_size)

        # Test with list
        xywh_list = [0.5, 0.5, 0.2, 0.2]
        result_list = xywh_to_xyxy(xywh_list, image_size)

        assert result_tuple == result_list
