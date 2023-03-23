import numpy as np
import torch
from sklearn.model_selection import train_test_split

from modules import helper
from modules import models


def save_model(model, model_path: str) -> None:
    return torch.save(model.state_dict(), model_path)


def initialise_model(model_name):
    model_object = getattr(models, model_name)
    return model_object


def load_model(model_object, model_path, n_features, z_dim):
    device = helper.get_device()
    model = model_object(n_features, z_dim)
    model.to(device)

    # Loading the state_dict into the model
    model.load_state_dict(torch.load(str(model_path)), strict=False)
    return model


def find_minmax(data):
    data = list(data)
    true_max_list = np.apply_along_axis(np.max, axis=0, arr=data)
    true_min_list = np.apply_along_axis(np.min, axis=0, arr=data)

    feature_range_list = true_max_list - true_min_list

    normalization_features = np.array([true_min_list, feature_range_list])
    return normalization_features


def normalize(data, custom_norm):
    data = np.array(data)
    if custom_norm:
        pass
    elif not custom_norm:
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
