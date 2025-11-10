"""Test cases for keypoint conversion helper functions.

Tests for:
- keypoints_to_numpy
- numpy_to_keypoints_flat
"""

import numpy as np
import pytest

from action_labeler.detections.helpers import (
    keypoints_to_numpy,
    numpy_to_keypoints_flat,
)


class TestKeypointsToNumpy:
    """Test cases for keypoints_to_numpy conversion."""

    def test_single_keypoint(self):
        """Test conversion of a single keypoint."""
        keypoints_flat = [0.5, 0.6]
        num_keypoints = 1

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([[0.5, 0.6]])
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (1, 2)

    def test_two_keypoints(self):
        """Test conversion of two keypoints."""
        keypoints_flat = [0.3, 0.4, 0.7, 0.8]
        num_keypoints = 2

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [0.3, 0.4],
            [0.7, 0.8],
        ])
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (2, 2)

    def test_three_keypoints(self):
        """Test conversion of three keypoints."""
        keypoints_flat = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        num_keypoints = 3

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ])
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (3, 2)

    def test_coco_17_keypoints(self):
        """Test conversion of COCO format with 17 keypoints."""
        # Create flat list with 34 values (17 keypoints * 2)
        keypoints_flat = [i * 0.05 for i in range(34)]
        num_keypoints = 17

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        assert result.shape == (17, 2)
        # Verify first and last keypoint
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.05)
        assert result[16, 0] == pytest.approx(1.6)
        assert result[16, 1] == pytest.approx(1.65)

    def test_zero_coordinates(self):
        """Test conversion with zero coordinates."""
        keypoints_flat = [0.0, 0.0, 0.0, 0.0]
        num_keypoints = 2

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_boundary_values(self):
        """Test conversion with boundary values (0 and 1)."""
        keypoints_flat = [0.0, 0.0, 1.0, 1.0, 0.5, 0.5]
        num_keypoints = 3

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_floating_point_precision(self):
        """Test that floating point precision is maintained."""
        keypoints_flat = [0.123456789, 0.987654321]
        num_keypoints = 1

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        assert abs(result[0, 0] - 0.123456789) < 1e-10
        assert abs(result[0, 1] - 0.987654321) < 1e-10

    def test_negative_values(self):
        """Test conversion with negative values (edge case - invalid but parseable)."""
        keypoints_flat = [-0.1, -0.2, 0.5, 0.6]
        num_keypoints = 2

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [-0.1, -0.2],
            [0.5, 0.6],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_values_greater_than_one(self):
        """Test conversion with values >1.0 (edge case - invalid but parseable)."""
        keypoints_flat = [1.5, 2.0, 0.5, 0.6]
        num_keypoints = 2

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [1.5, 2.0],
            [0.5, 0.6],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_very_small_values(self):
        """Test conversion with very small values."""
        keypoints_flat = [0.001, 0.002, 0.003, 0.004]
        num_keypoints = 2

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [0.001, 0.002],
            [0.003, 0.004],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_scientific_notation_values(self):
        """Test conversion with values in scientific notation."""
        keypoints_flat = [1e-3, 2e-3, 3e-3, 4e-3]
        num_keypoints = 2

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [0.001, 0.002],
            [0.003, 0.004],
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_integer_values(self):
        """Test conversion with integer values."""
        keypoints_flat = [0, 1, 2, 3]
        num_keypoints = 2

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [0.0, 1.0],
            [2.0, 3.0],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_mixed_int_float_values(self):
        """Test conversion with mixed int/float values."""
        keypoints_flat = [0, 0.5, 1, 0.75]
        num_keypoints = 2

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        expected = np.array([
            [0.0, 0.5],
            [1.0, 0.75],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_returns_numpy_array(self):
        """Test that the function returns a numpy array."""
        keypoints_flat = [0.5, 0.6]
        num_keypoints = 1

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        assert isinstance(result, np.ndarray)
        assert result.dtype in [np.float64, np.float32]

    def test_incorrect_length_raises_error(self):
        """Test that incorrect length raises an error."""
        # 5 values cannot be evenly divided into 2 coordinates per keypoint
        keypoints_flat = [0.1, 0.2, 0.3, 0.4, 0.5]
        num_keypoints = 2

        with pytest.raises(ValueError):
            keypoints_to_numpy(keypoints_flat, num_keypoints)

    def test_empty_list_zero_keypoints(self):
        """Test conversion with empty list and zero keypoints."""
        keypoints_flat = []
        num_keypoints = 0

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        assert result.shape == (0, 2)

    def test_many_keypoints(self):
        """Test conversion with many keypoints."""
        num_keypoints = 100
        keypoints_flat = [i * 0.01 for i in range(num_keypoints * 2)]

        result = keypoints_to_numpy(keypoints_flat, num_keypoints)

        assert result.shape == (100, 2)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.01
        assert result[99, 0] == 1.98
        assert result[99, 1] == 1.99


class TestNumpyToKeypointsFlat:
    """Test cases for numpy_to_keypoints_flat conversion."""

    def test_single_keypoint(self):
        """Test conversion of a single keypoint."""
        keypoints = np.array([[0.5, 0.6]])

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [0.5, 0.6]

    def test_two_keypoints(self):
        """Test conversion of two keypoints."""
        keypoints = np.array([
            [0.3, 0.4],
            [0.7, 0.8],
        ])

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [0.3, 0.4, 0.7, 0.8]

    def test_three_keypoints(self):
        """Test conversion of three keypoints."""
        keypoints = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ])

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_coco_17_keypoints(self):
        """Test conversion of COCO format with 17 keypoints."""
        # Create array with 17 keypoints
        keypoints = np.array([[i * 0.05, i * 0.05 + 0.01] for i in range(17)])

        result = numpy_to_keypoints_flat(keypoints)

        assert len(result) == 34
        assert result[0] == 0.0
        assert result[1] == 0.01
        assert result[32] == 0.8
        assert abs(result[33] - 0.81) < 1e-10

    def test_zero_coordinates(self):
        """Test conversion with zero coordinates."""
        keypoints = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_boundary_values(self):
        """Test conversion with boundary values (0 and 1)."""
        keypoints = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [0.0, 0.0, 1.0, 1.0, 0.5, 0.5]

    def test_floating_point_precision(self):
        """Test that floating point precision is maintained."""
        keypoints = np.array([[0.123456789, 0.987654321]])

        result = numpy_to_keypoints_flat(keypoints)

        assert abs(result[0] - 0.123456789) < 1e-10
        assert abs(result[1] - 0.987654321) < 1e-10

    def test_negative_values(self):
        """Test conversion with negative values."""
        keypoints = np.array([
            [-0.1, -0.2],
            [0.5, 0.6],
        ])

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [-0.1, -0.2, 0.5, 0.6]

    def test_values_greater_than_one(self):
        """Test conversion with values >1.0."""
        keypoints = np.array([
            [1.5, 2.0],
            [0.5, 0.6],
        ])

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [1.5, 2.0, 0.5, 0.6]

    def test_returns_list(self):
        """Test that the function returns a list."""
        keypoints = np.array([[0.5, 0.6]])

        result = numpy_to_keypoints_flat(keypoints)

        assert isinstance(result, list)

    def test_empty_array(self):
        """Test conversion with empty array."""
        keypoints = np.array([]).reshape(0, 2)

        result = numpy_to_keypoints_flat(keypoints)

        assert result == []

    def test_many_keypoints(self):
        """Test conversion with many keypoints."""
        keypoints = np.array([[i * 0.01, i * 0.01 + 0.005] for i in range(100)])

        result = numpy_to_keypoints_flat(keypoints)

        assert len(result) == 200
        assert result[0] == 0.0
        assert result[1] == 0.005
        assert abs(result[198] - 0.99) < 1e-10
        assert abs(result[199] - 0.995) < 1e-10

    def test_integer_array(self):
        """Test conversion with integer numpy array."""
        keypoints = np.array([[0, 1], [2, 3]], dtype=np.int32)

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [0, 1, 2, 3]

    def test_float32_array(self):
        """Test conversion with float32 numpy array."""
        keypoints = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)

        result = numpy_to_keypoints_flat(keypoints)

        assert len(result) == 4
        assert abs(result[0] - 0.5) < 1e-6
        assert abs(result[1] - 0.6) < 1e-6

    def test_float64_array(self):
        """Test conversion with float64 numpy array."""
        keypoints = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)

        result = numpy_to_keypoints_flat(keypoints)

        assert result == [0.5, 0.6, 0.7, 0.8]


class TestRoundTripKeypointConversions:
    """Test that keypoint conversions are reversible."""

    def test_flat_to_numpy_to_flat(self):
        """Test round-trip conversion flat -> numpy -> flat."""
        original = [0.3, 0.4, 0.7, 0.8, 0.1, 0.2]
        num_keypoints = 3

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        assert result == original

    def test_numpy_to_flat_to_numpy(self):
        """Test round-trip conversion numpy -> flat -> numpy."""
        original = np.array([
            [0.3, 0.4],
            [0.7, 0.8],
            [0.1, 0.2],
        ])

        flat = numpy_to_keypoints_flat(original)
        result = keypoints_to_numpy(flat, 3)

        np.testing.assert_array_equal(result, original)

    def test_round_trip_single_keypoint(self):
        """Test round-trip for single keypoint."""
        original = [0.5, 0.6]
        num_keypoints = 1

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        assert result == original

    def test_round_trip_coco_17_keypoints(self):
        """Test round-trip for COCO 17 keypoints."""
        original = [i * 0.05 for i in range(34)]
        num_keypoints = 17

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        assert result == original

    def test_round_trip_many_keypoints(self):
        """Test round-trip for many keypoints."""
        original = [i * 0.01 for i in range(200)]
        num_keypoints = 100

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        assert result == original

    def test_round_trip_preserves_precision(self):
        """Test that round-trip preserves floating point precision."""
        original = [0.123456789, 0.987654321, 0.333333333, 0.666666666]
        num_keypoints = 2

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        for orig, res in zip(original, result):
            assert abs(orig - res) < 1e-10

    def test_round_trip_zero_keypoints(self):
        """Test round-trip with zero keypoints."""
        original = []
        num_keypoints = 0

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        assert result == original

    def test_round_trip_boundary_values(self):
        """Test round-trip with boundary values."""
        original = [0.0, 0.0, 1.0, 1.0, 0.5, 0.5]
        num_keypoints = 3

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        assert result == original

    def test_round_trip_negative_values(self):
        """Test round-trip with negative values."""
        original = [-0.1, -0.2, 0.5, 0.6]
        num_keypoints = 2

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        assert result == original

    def test_round_trip_values_greater_than_one(self):
        """Test round-trip with values >1.0."""
        original = [1.5, 2.0, 0.5, 0.6]
        num_keypoints = 2

        numpy_kp = keypoints_to_numpy(original, num_keypoints)
        result = numpy_to_keypoints_flat(numpy_kp)

        assert result == original


class TestKeypointEdgeCases:
    """Test edge cases specific to keypoint conversions."""

    def test_reshape_with_wrong_dimensions(self):
        """Test that wrong dimensions raise an error."""
        # 7 values cannot be reshaped into (3, 2)
        keypoints_flat = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        num_keypoints = 3

        with pytest.raises(ValueError):
            keypoints_to_numpy(keypoints_flat, num_keypoints)

    def test_1d_array_conversion(self):
        """Test that 1D numpy array is handled correctly by flatten."""
        # This tests the numpy_to_keypoints_flat with already flat array
        keypoints = np.array([0.1, 0.2, 0.3, 0.4])

        result = numpy_to_keypoints_flat(keypoints.reshape(2, 2))

        assert result == [0.1, 0.2, 0.3, 0.4]

    def test_3d_array_input(self):
        """Test that 3D array is flattened correctly."""
        # Create a 3D array and see if flatten handles it
        keypoints = np.array([[[0.1, 0.2], [0.3, 0.4]]])

        result = numpy_to_keypoints_flat(keypoints.reshape(2, 2))

        assert result == [0.1, 0.2, 0.3, 0.4]

    def test_fortran_order_array(self):
        """Test that Fortran-ordered arrays are handled correctly."""
        keypoints = np.array([[0.1, 0.2], [0.3, 0.4]], order='F')

        result = numpy_to_keypoints_flat(keypoints)

        # Flatten should handle Fortran order correctly
        assert result == [0.1, 0.2, 0.3, 0.4]

    def test_non_contiguous_array(self):
        """Test that non-contiguous arrays are handled correctly."""
        # Create non-contiguous array through slicing
        large_array = np.array([[i, i+1] for i in range(10)])
        keypoints = large_array[::2]  # Non-contiguous

        result = numpy_to_keypoints_flat(keypoints)

        assert len(result) == 10  # 5 keypoints * 2 coords

    def test_view_vs_copy(self):
        """Test that modifications don't affect original data."""
        original_flat = [0.1, 0.2, 0.3, 0.4]
        numpy_kp = keypoints_to_numpy(original_flat, 2)

        # Modify the numpy array
        numpy_kp[0, 0] = 999.0

        # Original list should be unchanged
        assert original_flat == [0.1, 0.2, 0.3, 0.4]

    def test_data_type_preservation(self):
        """Test that data types are handled appropriately."""
        keypoints_int = np.array([[0, 1], [2, 3]], dtype=np.int32)
        result = numpy_to_keypoints_flat(keypoints_int)

        # Should convert to float/int in list
        assert isinstance(result[0], (int, np.integer))
