import os

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch


def get_mean_node_activations(input_dict: dict) -> dict:
    output_dict = {}
    for kk in input_dict:
        output_dict_layer = []
        for node in input_dict[kk].T:
            output_dict_layer.append(torch.mean(node).item())
        output_dict[kk] = output_dict_layer
    return output_dict


def dict_to_square_matrix(input_dict: dict) -> np.array:
    """Function changes an input dictionary into a square np.array. Adds NaNs when the dimension of a dict key is less than of the final square matrix.

    Args:
        input_dict (dict)

    Returns:
        square_matrix (np.array)
    """
    means_dict = get_mean_node_activations(input_dict)
    max_number_of_nodes = 0
    number_of_layers = len(input_dict)
    for kk in means_dict:
        if len(means_dict[kk]) > max_number_of_nodes:
            max_number_of_nodes = len(means_dict[kk])
    square_matrix = np.empty((number_of_layers, max_number_of_nodes))
    counter = 0
    for kk in input_dict:
        layer = np.array(means_dict[kk])
        if len(layer) == max_number_of_nodes:
            square_matrix[counter] = layer
        else:
            layer = np.append(
                layer, np.zeros(max_number_of_nodes - len(layer)) + np.nan
            )
            square_matrix[counter] = layer
        counter += 1
    return square_matrix


def plot(data: np.array, output_path: str) -> None:
    nodes_numbers = np.array([0, 50, 100, 200])
    fig, ax = plt.subplots()
    NAP = ax.imshow(
        data.T,
        cmap="RdBu_r",
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        norm=matplotlib.colors.CenteredNorm(),
    )
    colorbar = plt.colorbar(NAP)
    colorbar.set_label("Activation")
    ax.set_title("Neural Activation Pattern")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Number of nodes")
    xtick_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xtick_loc))
    ax.set_xticklabels(["", "en1", "en2", "en3", "de1", "de2", "de3", ""])
    ax.set_yticks(nodes_numbers)
    ax.figure.savefig(os.path.join(output_path, "diagnostics.pdf"))


def diagnose(input_path: str, output_path: str) -> None:
    input = np.load(input_path)
    plot(input, output_path)
    print(
        "Diagnostics saved as diagnostics.pdf in the diagnostics folder of your project."
    )
