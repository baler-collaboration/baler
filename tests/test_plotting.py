import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
import sys

baler_dir = os.path.join(os.path.dirname(__file__), "..", "baler")
modules_dir = os.path.join(baler_dir, "modules")
sys.path.append(modules_dir)
import plotting
from plotting import get_index_to_cut
from plotting import plot_box_and_whisker
from plotting import loss_plot
from dataclasses import dataclass
from unittest import mock


def test_get_index_to_cut():
    # Create a test array
    test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Test case where all values are above the threshold
    cut = 0
    column_index = 0
    expected_result = np.array([])
    result = get_index_to_cut(column_index, cut, test_array)
    print("Expected Output : ", expected_result)
    print("Actual Output: ", result)
    assert np.array_equal(result, expected_result)

    # Test case where some values are below the threshold
    cut = 5
    column_index = 1
    expected_result = np.array([0])
    result = get_index_to_cut(column_index, cut, test_array)
    print("Expected Output : ", expected_result)
    print("Actual Output: ", result)
    assert np.array_equal(result, expected_result)

    # Test case where all values are below the threshold
    cut = 10
    column_index = 2
    expected_result = np.array([0, 1, 2])
    result = get_index_to_cut(column_index, cut, test_array)
    print("Expected Output : ", expected_result)
    print("Actual Output: ", result)
    assert np.array_equal(result, expected_result)

    # Test case where the input array is empty
    cut = 1
    column_index = 0
    expected_result = np.array([])
    result = get_index_to_cut(column_index, cut, np.array([]))
    print("Expected Output : ", expected_result)
    print("Actual Output: ", result)
    assert np.array_equal(result, expected_result)


def test_plot_box_and_whisker():
    # Create sample data
    names = ["column1", "column2", "column3"]
    residual = [np.array([1, 2, 3]), np.array([2, 4, 6]), np.array([3, 6, 9])]
    pdf = mock.MagicMock()

    # Call the function
    plot_box_and_whisker(names, residual, pdf)

    # Check that the plot was saved
    assert pdf.savefig.called
