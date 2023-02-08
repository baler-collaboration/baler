import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from baler.modules import data_processing


def test_import_config_success():
    # Call the import_config function with the sample config file path
    config = {"Foo": "Bar", "Baz": 10}
    config_path = "tmp_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Call the import_config function with the sample config file path
    result = data_processing.import_config(config_path)

    # Assert that the result is equal to the expected config
    # This checks that the import_config function correctly loads the JSON file and returns the expected dictionary
    assert result == config

    # Clean up the sample config file
    os.remove(config_path)


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
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            pd.DataFrame({"True min": [1, 2, 3], "Feature Range": [6, 6, 6]}),
        ),
        (
            [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]],
            pd.DataFrame({"True min": [-7, -8, -9], "Feature Range": [6, 6, 6]}),
        ),
        (
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            pd.DataFrame({"True min": [0, 0, 0], "Feature Range": [2, 2, 2]}),
        ),
    ]


def test_find_minmax_success(minmax_test_data):
    for data, expected_result in minmax_test_data:
        result = data_processing.find_minmax(data)
        assert result.equals(expected_result)


def test_normalize():
    # Test data
    data = [1, 2, 3, 4, 5]

    # Test configuration 1
    config1 = {"custom_norm": False}
    expected_result1 = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Test configuration 2
    config2 = {"custom_norm": True}
    expected_result2 = np.array([1, 2, 3, 4, 5])

    # Test the normalize function with the test data and configuration 1
    result1 = data_processing.normalize(data, config1)
    np.testing.assert_almost_equal(result1, expected_result1)

    # Test the normalize function with the test data and configuration 2
    result2 = data_processing.normalize(data, config2)
    np.testing.assert_almost_equal(result2, expected_result2)


def test_split_success():
    # Test data
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})
    test_size = 0.2
    random_state = 42

    # Split the data using the split function
    train, test = data_processing.split(df, test_size, random_state)

    # Check that the size of the train and test sets is correct
    assert train.shape[0] + test.shape[0] == df.shape[0]
    assert abs(train.shape[0] / df.shape[0] - (1 - test_size)) < 1e-9

    # Check that the random state was set correctly
    train_split, test_split = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    np.testing.assert_array_equal(train.index, train_split.index)
    np.testing.assert_array_equal(test.index, test_split.index)


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


def test_get_columns():
    # Test data
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Get the columns using the get_columns function
    columns = data_processing.get_columns(df)

    # Check that the columns are correct
    expected_columns = ["col1", "col2"]
    assert columns == expected_columns
