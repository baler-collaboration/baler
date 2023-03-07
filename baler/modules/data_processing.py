import json
import pickle

import numpy as np
import pandas as pd
import torch
import uproot
from sklearn.model_selection import train_test_split

from modules import helper
from modules import models


def import_config(config_path: str) -> dict:
    try:
        with open(config_path, encoding="utf-8") as json_config:
            config = json.load(json_config)
        return config
    except FileNotFoundError:
        print(f"Config file not found at path: {config_path}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse config file at path {config_path}: {e}")


def save_model(model, model_path: str) -> None:
    return torch.save(model.state_dict(), model_path)


def initialise_model(config):
    model_name = config["model_name"]
    model_object = getattr(models, model_name)
    return model_object


def load_model(model_object, model_path, n_features, z_dim):
    device = helper.get_device()
    model = model_object(device, n_features, z_dim)

    # Loading the state_dict into the model
    model.load_state_dict(torch.load(str(model_path)), strict=False)
    return model


def type_clearing(tt_tree):
    type_names = tt_tree.typenames()
    column_type = []
    column_names = []

    # In order to remove non integers or -floats in the TTree,
    # we separate the values and keys
    for keys in type_names:
        column_type.append(type_names[keys])
        column_names.append(keys)

    # Checks each value of the typename values to see if it isn't an int or
    # float, and then removes it
    for i in range(len(column_type)):
        if column_type[i] != "float[]" and column_type[i] != "int32_t[]":
            # print('Index ',i,' was of type ',Typename_list_values[i],'\
            # and was deleted from the file')
            del column_names[i]

    # Returns list of column names to use in load_data function
    return column_names


def numpy_to_df(array, config):
    if np.shape(array)[1] == 4:
        col_names = ["comp1", "comp2", "comp3", "comp4"]
    else:
        col_names = config["cleared_col_names"]
    df = pd.DataFrame(array, columns=col_names)

    return df


def load_data(data_path: str, config):
    file_extension = data_path.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(data_path, low_memory=False)
    elif file_extension == "root":
        tree = uproot.open(data_path)[config["Branch"]][config["Collection"]][
            config["Objects"]
        ]
        global names
        names = type_clearing(tree)
        df = tree.arrays(names, library="pd")
    elif file_extension == "pickle" or file_extension == "pkl":
        df = pd.read_pickle(data_path)
    else:
        raise Exception(f"File extension {file_extension} not supported")

    return df


def clean_data(df, config):
    df = df.drop(columns=config["dropped_variables"])
    df = df.dropna()
    global cleared_column_names
    cleared_column_names = list(df)
    return df


def find_minmax(data):
    data = np.array(data)
    data = list(data)
    true_max_list = np.apply_along_axis(np.max, axis=0, arr=data)
    true_min_list = np.apply_along_axis(np.min, axis=0, arr=data)

    feature_range_list = true_max_list - true_min_list

    normalization_features = pd.DataFrame(
        {"True min": true_min_list, "Feature Range": feature_range_list}
    )
    return normalization_features


def normalize(data, config):
    data = np.array(data)
    if config["custom_norm"] is True:
        pass
    elif config["custom_norm"] is False:
        true_min = np.min(data)
        true_max = np.max(data)
        feature_range = true_max - true_min
        data = [((i - true_min) / feature_range) for i in data]
        data = np.array(data)
    return data


def split(df, test_size, random_state):
    return train_test_split(df, test_size=test_size, random_state=random_state)


def renormalize_std(data, true_min, feature_range):
    data = list(data)
    data = [((i * feature_range) + true_min) for i in data]
    data = np.array(data)
    return data


def renormalize_func(norm_data, min_list, range_list):
    norm_data = np.array(norm_data)
    renormalized = [
        renormalize_std(norm_data, min_list[i], range_list[i])
        for i in range(len(min_list))
    ]
    renormalized_full = [(renormalized[i][:, i]) for i in range(len(renormalized))]
    renormalized_full = np.array(renormalized_full).T
    return renormalized_full


def get_columns(df):
    return list(df.columns)


def pickle_to_df(file_path, config):
    load_data(file_path, config)
    # From pickle to df:
    with open(file_path, "rb") as handle:
        data = pickle.load(handle)
        df = pd.DataFrame(data, columns=names)
        return df, names


def df_to_root(df, col_names, save_path):
    col = col_names
    data = np.array(df)
    Event_dict = {}
    Value_dict = {}

    # Fills a dictionary with Branch names and which dtype to fill said branch with
    for Branch in col:
        Event_dict[Branch] = np.float64

    # Fills a dictionary with Branch names and corresponding value
    for i in range(len(col)):
        Value_dict[col[i]] = data[:, i]

    # Creates the root file
    with uproot.recreate(save_path) as fout:
        # Creates the tree & branches
        fout.mktree(
            "Events",
            Event_dict,
        )
        # Fills the branches
        fout["Events"].extend(Value_dict)


def RMS_function(response_norm):
    square = np.square(response_norm)
    MS = square.mean()
    RMS = np.sqrt(MS)
    return RMS
