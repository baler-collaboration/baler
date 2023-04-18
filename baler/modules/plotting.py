# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from tqdm import trange
import os



def loss_plot(path_to_loss_data, output_path, config):
    """This function Plots the loss from the training and saves it

    Args:
        path_to_loss_data (string): Path to file containing loss plot data generated during training
        output_path (path): Directory path to which the loss plot is saved
        config (dataclass): The config class containing attributes set in the config file
    """
    loss_data = np.load(path_to_loss_data)
    str_list = ["Epochs:", "Model Name:", "Reg. Param:", "lr:", "BS:"]

    train_loss = loss_data[0]
    val_loss = loss_data[1]
    conf_list = [
        len(train_loss),
        config.model_name,
        config.reg_param,
        config.lr,
        config.batch_size,
    ]

    plt.figure(figsize=(10, 7))
    plt.title("Loss plot")
    plt.plot(train_loss, color="orange", label="Train Loss")
    if config.test_size:
        plt.plot(val_loss, color="red", label="Validation Loss")
    for i in range(len(conf_list)):
        plt.plot([], [], " ", label=str_list[i] + " " + str(conf_list[i]))
    plt.xlabel("Epochs")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_path, "plotting", "Loss_plot.pdf"))
    # plt.show()


def get_index_to_cut(column_index, cut, array):
    """Given an array column index and a threshold, this function returns the index of the
        entries not passing the threshold.

    Args:
        column_index (int): The index for the column where cuts should be applied
        cut (float): Threshold for which values below will have the whole entry removed
        array (np.array): The full array to be edited

    Returns:
        _type_: returns the index of the rows to be removed
    """
    indices_to_cut = np.argwhere(array[column_index] < cut).flatten()
    return indices_to_cut


def plot_box_and_whisker(names, residual, pdf):
    """Plots Box and Whisker plots of 1D data

    Args:
        project_path (string): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
    """
    column_names = [i.split(".")[-1] for i in names]

    fig1, ax1 = plt.subplots()

    boxes = ax1.boxplot(list(residual), showfliers=False, vert=False)
    whiskers = np.concatenate([item.get_xdata() for item in boxes["whiskers"]])
    edges = max([abs(min(whiskers)), max(whiskers)])

    ax1.set_yticks(np.arange(1, len(column_names) + 1, 1))
    ax1.set_yticklabels(column_names)

    ax1.grid()
    fig1.tight_layout()
    ax1.set_xlabel("Residual")
    ax1.set_xlim(-edges - edges * 0.1, edges + edges * 0.1)
    pdf.savefig()


def plot_1D(output_path: str, config):
    """General plotting for 1D data, for example data from a '.csv' file. This function generates a pdf
        document where each page contains the before/after performance
        of each column of the 1D data

    Args:
        output_path (path): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
    """

    before_path = config.input_path
    after_path = os.path.join(output_path, "decompressed_output", "decompressed.npz")

    before = np.transpose(np.load(before_path)["data"])
    after = np.transpose(np.load(after_path)["data"])
    names = np.load(config.input_path)["names"]

    index_to_cut = get_index_to_cut(3, 1e-6, before)
    before = np.delete(before, index_to_cut, axis=1)
    after = np.delete(after, index_to_cut, axis=1)

    response = np.divide(np.subtract(after, before), before) * 100
    residual = np.subtract(after, before)

    with PdfPages(os.path.join(output_path, "plotting", "comparison.pdf")) as pdf:
        plot_box_and_whisker(names, residual, pdf)
        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1])

        axsLeft = subfigs[0].subplots(2, 1, sharex=True)
        ax1 = axsLeft[0]
        ax3 = axsLeft[1]
        axsRight = subfigs[1].subplots(2, 1, sharex=False)
        ax2 = axsRight[0]
        ax4 = axsRight[1]

        number_of_columns = len(names)

        print("=== Plotting ===")

        for index, column in enumerate(tqdm(names)):
            column_name = column.split(".")[-1]
            rms = np.sqrt(np.mean(np.square(response[index])))
            residual_RMS = np.sqrt(np.mean(np.square(residual[index])))

            x_min = min(before[index] + after[index])
            x_max = max(before[index] + after[index])
            x_diff = abs(x_max - x_min)

            # Before Histogram
            counts_before, bins_before = np.histogram(
                before[index],
                bins=np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, 200),
            )
            ax1.hist(
                bins_before[:-1], bins_before, weights=counts_before, label="Before"
            )

            # After Histogram
            counts_after, bins_after = np.histogram(
                after[index],
                bins=np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, 200),
            )
            ax1.hist(
                bins_after[:-1],
                bins_after,
                weights=counts_after,
                label="After",
                histtype="step",
            )

            ax1.set_ylabel("Counts", ha="right", y=1.0)
            ax1.set_yscale("log")
            ax1.legend(loc="best")
            ax1.set_xlim(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff)
            ax1.set_ylim(ymin=1)

            data_bin_centers = bins_after[:-1] + (bins_after[1:] - bins_after[:-1]) / 2
            ax3.scatter(
                data_bin_centers, (counts_after - counts_before), marker="."
            )  # FIXME: Dividing by zero
            ax3.axhline(y=0, linewidth=0.2, color="black")
            ax3.set_xlabel(f"{column_name}", ha="right", x=1.0)
            ax3.set_ylim(
                -max(counts_after - counts_before)
                - 0.05 * max(counts_after - counts_before),
                max(counts_after - counts_before)
                + 0.05 * max(counts_after - counts_before),
            )
            ax3.set_ylabel("Residual")

            # Response Histogram
            counts_response, bins_response = np.histogram(
                response[index], bins=np.arange(-20, 20, 0.1)
            )
            ax2.hist(
                bins_response[:-1],
                bins_response,
                weights=counts_response,
                label="Response",
            )
            ax2.axvline(
                np.mean(response[index]),
                color="k",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean {round(np.mean(response[index]),4)} %",
            )
            ax2.plot([], [], " ", label=f"RMS: {round(rms,4)} %")

            ax2.set_xlabel(f"{column_name} Response [%]", ha="right", x=1.0)
            ax2.set_ylabel("Counts", ha="right", y=1.0)
            ax2.legend(loc="best", bbox_to_anchor=(1, 1.05))

            # Residual Histogram
            counts_residual, bins_residual = np.histogram(
                residual[index], bins=np.arange(-1, 1, 0.01)
            )
            ax4.hist(
                bins_residual[:-1],
                bins_residual,
                weights=counts_residual,
                label="Residual",
            )
            ax4.axvline(
                np.mean(residual[index]),
                color="k",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean {round(np.mean(residual[index]),6)}",
            )
            ax4.plot([], [], " ", label=f"RMS: {round(residual_RMS,6)}")
            ax4.plot([], [], " ", label=f"Max: {round(max(residual[index]),6)}")
            ax4.plot([], [], " ", label=f"Min: {round(min(residual[index]),6)}")

            ax4.set_xlabel(f"{column_name} Residual", ha="right", x=1.0)
            ax4.set_ylabel("Counts", ha="right", y=1.0)
            ax4.set_xlim(-1, 1)
            ax4.legend(loc="best", bbox_to_anchor=(1, 1.05))

            pdf.savefig()
            ax2.clear()
            ax1.clear()
            ax3.clear()
            ax4.clear()


def plot_2D(output_path, config):
    """General plotting for 2D data, for example 2D arraysfrom computational fluid
        dynamics or other image like data. This function generates a pdf
        document where each page contains the before/after performance
        of each column of the 1D data

    Args:
        output_path (path): The path to the output directory
        config (dataclass): The config class containing attributes set in the config file
    """

    data = np.load(config.input_path)["data"]
    data_decompressed = np.load(
        os.path.join(output_path, "decompressed_output", "decompressed.npz")
    )["data"]

    if data.shape[0] > 1:
        num_tiles = data.shape[0]
    else:
        num_tiles = 1

    print("=== Plotting ===")

    for ind in trange(num_tiles):
        tile_data_decompressed = data_decompressed[ind].reshape(
            data_decompressed.shape[2], data_decompressed.shape[3]
        )
        tile_data = data[ind].reshape(data.shape[1], data.shape[2])

        diff = ((tile_data_decompressed - tile_data) / tile_data) * 100

        fig, axs = plt.subplots(
            1, 3, figsize=(29.7 * (1 / 2.54), 21 * (1 / 2.54)), sharey=True
        )
        axs[0].set_title("Original", fontsize=11, y=-0.2)
        im1 = axs[0].imshow(
            tile_data, vmin=-0.01, vmax=0.07, cmap="CMRmap", interpolation="nearest"
        )
        cb2 = plt.colorbar(im1, ax=[axs[0]], location="top")

        axs[1].set_title("Decompressed", fontsize=11, y=-0.2)
        im2 = axs[1].imshow(
            tile_data_decompressed,
            vmin=-0.01,
            vmax=0.07,
            cmap="CMRmap",
            interpolation="nearest",
        )
        cb2 = plt.colorbar(im2, ax=[axs[1]], location="top")

        axs[2].set_title("Relative Diff. [%]", fontsize=11, y=-0.2)
        im3 = axs[2].imshow(
            diff, vmin=-50, vmax=50, cmap="cool_r", interpolation="nearest"
        )
        cb2 = plt.colorbar(im3, ax=[axs[2]], location="top")

        plt.ylim(0, 50)
        plt.xlim(0, 50)
        fig.suptitle(
            "Compressed file is 10% the size of original,\n100 epochs (4.52 min)",
            y=0.9,
            fontsize=16,
        )

        fig.savefig(
            os.path.join(output_path, "plotting", f"CFD{ind}.jpg"), bbox_inches="tight"
        )


def plot(output_path, config):
    """Runs the appropriate plotting function based on the data dimension 1D or 2D

    Args:
        output_path (path): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
    """
    if config.data_dimension == 1:
        plot_1D(output_path, config)
    elif config.data_dimension == 2:
        plot_2D(output_path, config)
