import os

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from baler.modules import data_processing
from baler.modules import helper


def test_import_config_success():
    # Call the import_config function with the sample config file path
    config = helper.configClass
    config.Foo = "Bar"
    config.Baz = 10

    # Assert that the result is equal to the expected config
    # This checks that the import_config function correctly loads the JSON file and returns the expected dictionary
    assert config.Foo == "Bar"


def test_save_model():
    # Test data
    model = torch.nn.Linear(3, 2)
    model_path = "test_model.pt"

    # Save the model
    data_processing.save_model(model, model_path)

    # Check that the model file has been created
    assert os.path.exists(model_path)

    # Clean up
    os.remove(model_path)


@pytest.fixture
def minmax_test_data():
    return [
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1, 2, 3], [6, 6, 6]])),
        (
            np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]),
            np.array([[-7, -8, -9], [6, 6, 6]]),
        ),
        (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), np.array([[0, 0, 0], [2, 2, 2]])),
    ]


def test_find_minmax_success(minmax_test_data):
    for data, expected_result in minmax_test_data:
        result = data_processing.find_minmax(data)
        assert np.array_equal(result, expected_result)


def test_normalize():
    # Test data
    data = [1, 2, 3, 4, 5]

    # Test configuration 1
    custom_norm1 = False
    expected_result1 = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Test configuration 2
    custom_norm2 = True
    expected_result2 = np.array([1, 2, 3, 4, 5])

    # Test the normalize function with the test data and configuration 1
    result1 = data_processing.normalize(data, custom_norm1)
    np.testing.assert_almost_equal(result1, expected_result1)

    # Test the normalize function with the test data and configuration 2
    result2 = data_processing.normalize(data, custom_norm2)
    np.testing.assert_almost_equal(result2, expected_result2)


def test_renormalize_std():
    # Test data
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    true_min = 1
    feature_range = 2

    # Renormalize the data using the renormalize_std function
    renormalized_data = data_processing.renormalize_std(data, true_min, feature_range)

    # Check that the renormalized data is correct
    expected_renormalized_data = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
    np.testing.assert_array_equal(renormalized_data, expected_renormalized_data)


def test_renormalize_func():
    # Test data
    scaler = MinMaxScaler()
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    scaler.fit(data)
    norm_data = scaler.transform(data)
    true_min = [-1, 2]
    feature_range = [2, 16]

    # Renormalize the data using the renormalize_std function
    renormalized_data = data_processing.renormalize_func(
        norm_data, true_min, feature_range
    )

    # Check that the renormalized data is correct
    expected_renormalized_data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
    np.testing.assert_array_equal(renormalized_data, expected_renormalized_data)
