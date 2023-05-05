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
from tqdm import tqdm
from tqdm import trange
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import seaborn as sns
import pandas as pd
import sys


def loss_plot(path_to_loss_data, output_path, config):
    """This function Plots the loss from the training and saves it

    Args:
        path_to_loss_data (string): Path to file containing loss plot data generated during training
        output_path (string): Directory path to which the loss plot is saved
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
    plt.savefig(output_path + "_Loss_plot.pdf")
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


def fiol(names, residual, pdf):
    fig1, ax1 = plt.subplots()
    residual = np.transpose(residual)
    print(names)
    df = pd.DataFrame(residual, columns=names)
    sns.violinplot(x=df["recoPFJets_ak5PFJets__RECO.obj.pt_"], inner=None)
    ax1.set_xlim(-0.5, 0.5)
    pdf.savefig()
    sys.exit()


def alex(project_path, names, before, after):
    fig, axis_list = plt.subplots(
        6, 4, constrained_layout=True, figsize=(11.69, 8.27), sharey=True
    )
    second_axis_list = axis_list

    with PdfPages(project_path + "/plotting/alex.pdf") as pdf:
        print(axis_list.shape)
        for column_index1, column in enumerate(tqdm(names)):
            axis_column = column_index1 % 4
            axis_row = column_index1 // 6
            print(axis_row, axis_column)

            row_index = axis_row
            column_index = axis_column

            second_axis_list[row_index][column_index] = axis_list[row_index][
                column_index
            ].twinx()
            column_name = column.split(".")[-1]

            x_min = min(before[column_index] + after[column_index])
            x_max = max(before[column_index] + after[column_index])
            x_diff = abs(x_max - x_min)

            # Before Histogram
            counts_before, bins_before = np.histogram(
                before[column_index],
                bins=np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, 200),
            )
            axis_list[row_index][column_index].hist(
                bins_before[:-1], bins_before, weights=counts_before, label="Before"
            )

            # After Histogram
            counts_after, bins_after = np.histogram(
                after[column_index],
                bins=np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, 200),
            )
            axis_list[row_index][row_index].hist(
                bins_after[:-1],
                bins_after,
                weights=counts_after,
                label="After",
                histtype="step",
            )

            axis_list[row_index][row_index].set_ylabel("Counts", ha="right", y=1.0)
            axis_list[row_index][row_index].set_yscale("log")
            axis_list[row_index][row_index].legend(loc="best")
            axis_list[row_index][row_index].set_xlim(
                x_min - 0.1 * x_diff, x_max + 0.1 * x_diff
            )
            axis_list[row_index][row_index].set_ylim(ymin=1)

            data_bin_centers = bins_after[:-1] + (bins_after[1:] - bins_after[:-1]) / 2
            second_axis_list[row_index][row_index].scatter(
                data_bin_centers,
                (np.abs((counts_before - counts_after)) / counts_after) * 100,
                marker=".",
                color="black",
            )  # FIXME: Dividing by zero
            second_axis_list[row_index][row_index].set_ylim(0, 100)
            second_axis_list[row_index][row_index].set_ylabel(
                "Relative Difference [%]", x=-1
            )

        pdf.savefig()


def plot_1D(project_path, config):
    """General plotting for 1D data, for example data from a '.csv' file. This function generates a pdf
        document where each page contains the before/after performance
        of each column of the 1D data

    Args:
        project_path (string): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
    """

    output_path = project_path + "training/"
    before_path = config.input_path
    after_path = project_path + "decompressed_output/decompressed.npz"

    before = np.transpose(np.load(before_path)["data"])
    after = np.transpose(np.load(after_path)["data"])
    names = np.load(config.input_path)["names"]

    index_to_cut = get_index_to_cut(3, 1e-6, before)
    before = np.delete(before, index_to_cut, axis=1)
    after = np.delete(after, index_to_cut, axis=1)

    response = np.divide(np.subtract(after, before), before) * 100
    residual = np.subtract(after, before)

    alex(project_path, names, before, after)

    with PdfPages(project_path + "/plotting/comparison.pdf") as pdf:
        # fiol(names, residual, pdf)
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
                response[index], bins=np.arange(-20, 20, 0.2)
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


def getIndex(cf, ci, d):
    "Converts the dimensionless coordinates to cell indices"
    index = int((cf - ci) / d + 0.5)
    return index


def animate(t):
    cs = plt.contourf(CFD_data[t, 0, :, :], cmap=cm, levels=50)
    # plt.colorbar(cs, orientation = 'horizontal')
    # plt.gca().set_aspect('equal')


def plot_2D(project_path, config):
    """General plotting for 2D data, for example 2D arraysfrom computational fluid
        dynamics or other image like data. This function generates a pdf
        document where each page contains the before/after performance
        of each column of the 1D data

    Args:
        project_path (string): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
    """

    data = np.load(config.input_path)["data"]
    data_decompressed = np.load(project_path + "/decompressed_output/decompressed.npz")[
        "data"
    ]

    if data.shape[0] > 1:
        num_tiles = data.shape[0]
    else:
        num_tiles = 1

    print("=== Plotting ===")

    for ind in trange(num_tiles):
        tile_data_decompressed = data_decompressed[ind][0] * 0.04 * 1000
        tile_data = data[ind] * 0.04 * 1000

        diff = tile_data - tile_data_decompressed

        fig, axs = plt.subplots(
            1, 3, figsize=(29.7 * (1 / 2.54), 10 * (1 / 2.54)), sharey=True
        )
        axs[0].set_title("Original", fontsize=11)
        im1 = axs[0].imshow(
            tile_data,
            vmin=-0.5,
            vmax=3.0,
            cmap="CMRmap",
            interpolation="nearest",
        )
        axis = axs[0]
        axis.tick_params(axis="both", which="major")
        plt.ylim(0, 50)
        plt.xlim(0, 50)
        axis.set_ylabel("y [m]")
        axis.set_xlabel("x [m]")
        axis.set_xticks([10, 20, 30, 40, 50])
        axis.set_xticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
        axis.set_yticks([10, 20, 30, 40, 50])
        axis.set_yticklabels([0.4, 0.8, 1.2, 1.6, 2.0])

        axs[1].set_title("Reconstructed", fontsize=11)
        im2 = axs[1].imshow(
            tile_data_decompressed,
            vmin=-0.5,
            vmax=3.0,
            cmap="CMRmap",
            interpolation="nearest",
        )
        axis = axs[1]
        axis.tick_params(axis="both", which="major")
        plt.ylim(0, 50)
        plt.xlim(0, 50)
        axis.set_ylabel("y [m]")
        axis.set_xlabel("x [m]")
        axis.set_xticks([10, 20, 30, 40, 50])
        axis.set_xticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
        axis.set_yticks([10, 20, 30, 40, 50])
        axis.set_yticklabels([0.4, 0.8, 1.2, 1.6, 2.0])

        axs[2].set_title("Difference", fontsize=11)
        im3 = axs[2].imshow(
            diff,
            vmin=-0.5,
            vmax=3.0,
            cmap="CMRmap",
            interpolation="nearest",
        )
        # cb2 = plt.colorbar(im3, ax=[axs[2]], location="right", fraction=0.046, pad=0.1)
        # cb2.set_label("x-velocity [mm/s]")
        axis = axs[2]
        axis.tick_params(axis="both", which="major")
        plt.ylim(0, 50)
        plt.xlim(0, 50)
        axis.set_ylabel("y [m]")
        axis.set_xlabel("x [m]")
        axis.set_xticks([10, 20, 30, 40, 50])
        axis.set_xticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
        axis.set_yticks([10, 20, 30, 40, 50])
        axis.set_yticklabels([0.4, 0.8, 1.2, 1.6, 2.0])

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.815, 0.2, 0.02, 0.59])
        cb2 = fig.colorbar(im3, cax=cbar_ax, location="right", aspect=10)
        cb2.set_label("x-velocity [m/s]")
        # fig.colorbar(im3, cax=cbar_ax)

        fig.savefig(
            project_path + "/plotting/CFD" + str(ind) + ".png", bbox_inches="tight"
        )
        # sys.exit()

    # import imageio.v2 as imageio

    # with imageio.get_writer(project_path + "/plotting/CFD.gif", mode="I") as writer:
    #     for i in range(0, 60):
    #         path = project_path + "/plotting/CFD" + str(i) + ".jpg"
    #         print(path)
    #         image = imageio.imread(path)
    #         writer.append_data(image)


# def plot_2D(project_path, config):
#     # Selects time steps of a 2D GASCANS CFD simulation and stores them in a .npz file ready for Baler.

#     # import numpy as np
#     # import matplotlib.pyplot as plt
#     # import h5py
#     # from matplotlib.animation import FuncAnimation

#     # # Change font size
#     # plt.rcParams.update({"font.size": 8})

#     # colours = "b", "g", "orange"
#     # lines = "solid", "dotted", "dashed", "dashdot"
#     # cm = "magma_r"

#     # #### GASCANS CFD results ####
#     # data_path = project_path + "decompressed_output//"
#     # fname = "decompressed.npz"

#     # start_time = 30000
#     # time_step = 100
#     # num_steps = 60

#     # baler_data = np.load(data_path + fname)
#     # CFD_data = baler_data["data"]

#     # # Show animation
#     # fig, ax = plt.subplots()
#     # ani = FuncAnimation(fig, animate, frames=num_steps, interval=50, repeat=False)
#     # ani.save("CFDAnimation.gif", writer="imagemagick", fps=20)
#     # plt.show()

#     """General plotting for 2D data, for example 2D arraysfrom computational fluid
#         dynamics or other image like data. This function generates a pdf
#         document where each page contains the before/after performance
#         of each column of the 1D data

#     Args:
#         project_path (string): The path to the project directory
#         config (dataclass): The config class containing attributes set in the config file
#     """

#     data = np.load(config.input_path)["data"]
#     data_decompressed = np.load(project_path + "/decompressed_output/decompressed.npz")[
#         "data"
#     ]

#     if data.shape[0] > 1:
#         num_tiles = data.shape[0]
#     else:
#         num_tiles = 1

#     print("=== Plotting ===")

#     for ind in trange(num_tiles):
#         tile_data_decompressed = data_decompressed[ind].reshape(
#             data_decompressed.shape[2], data_decompressed.shape[3]
#         )
#         tile_data = np.flip(data[ind].reshape(data.shape[1], data.shape[2]))

#         tile_data_decompressed = tile_data_decompressed * 0.025 * 1000
#         tile_data = np.flip(tile_data) * 0.025 * 1000
#         print(tile_data)

#         # diff = ((tile_data_decompressed - tile_data) / tile_data_decompressed) * 100
#         diff = tile_data - tile_data_decompressed

#         fig, axs = plt.subplots(
#             figsize=(29.7 * (1 / 2.54), 21 * (1 / 2.54)), sharey=True
#         )
#         axs.set_title("Before", fontsize=28)
#         im1 = axs.imshow(
#             tile_data,
#             vmin=-1 / 4,
#             vmax=7 / 4,
#             cmap="CMRmap",
#             interpolation="nearest",
#         )
#         cb2 = plt.colorbar(im1, ax=[axs], location="right")
#         cb2.ax.tick_params(labelsize=22)
#         cb2.ax.yaxis.offsetText.set_fontsize(22)
#         cb2.set_label("x-velocity [mm/s]", fontsize=28)
#         axs.tick_params(axis="both", which="major", labelsize=22)
#         plt.ylim(0, 50)
#         plt.xlim(0, 50)
#         axs.set_ylabel("y [m]", fontsize=28)
#         axs.set_xlabel("x [m]", fontsize=28)
#         axs.set_xticks([10, 20, 30, 40, 50])
#         axs.set_xticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
#         axs.set_yticks([10, 20, 30, 40, 50])
#         axs.set_yticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
#         fig.savefig(project_path + "/plotting/CFD_before" + ".pdf", bbox_inches="tight")
#         cb2.ax.tick_params(labelsize=22)
#         cb2.ax.yaxis.offsetText.set_fontsize(22)

#         fig, axs = plt.subplots(
#             figsize=(29.7 * (1 / 2.54), 21 * (1 / 2.54)), sharey=True
#         )
#         axs.set_title("After", fontsize=28)
#         im2 = axs.imshow(
#             tile_data_decompressed,
#             vmin=-1 / 4,
#             vmax=7 / 4,
#             cmap="CMRmap",
#             interpolation="nearest",
#         )
#         cb2 = plt.colorbar(im2, ax=[axs], location="right")
#         cb2.ax.tick_params(labelsize=22)
#         cb2.ax.yaxis.offsetText.set_fontsize(22)
#         cb2.ax.tick_params(labelsize=22)
#         cb2.ax.yaxis.offsetText.set_fontsize(22)
#         cb2.set_label("x-velocity [mm/s]", fontsize=28)
#         axs.tick_params(axis="both", which="major", labelsize=22)
#         plt.ylim(0, 50)
#         plt.xlim(0, 50)
#         axs.set_ylabel("y [m]", fontsize=28)
#         axs.set_xlabel("x [m]", fontsize=28)
#         axs.set_xticks([10, 20, 30, 40, 50])
#         axs.set_xticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
#         axs.set_yticks([10, 20, 30, 40, 50])
#         axs.set_yticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
#         fig.savefig(
#             project_path + "/plotting/CFD_decompressed" + ".pdf", bbox_inches="tight"
#         )

#         fig, axs = plt.subplots(
#             figsize=(29.7 * (1 / 2.54), 21 * (1 / 2.54)), sharey=True
#         )
#         axs.set_title("Residual = Before - After", fontsize=28)
#         im3 = axs.imshow(
#             diff,
#             vmin=-0.00015 / 4,
#             vmax=0.00015 / 4,
#             cmap="cool_r",
#             interpolation="nearest",
#         )
#         cb2 = plt.colorbar(im3, ax=[axs], location="right")
#         cb2.ax.tick_params(labelsize=22)
#         cb2.ax.yaxis.offsetText.set_fontsize(18)
#         cb2.set_label("x-velocity [mm/s]", fontsize=28)
#         axs.tick_params(axis="both", which="major", labelsize=22)
#         plt.ylim(0, 50)
#         plt.xlim(0, 50)
#         axs.set_ylabel("y [m]", fontsize=28)
#         axs.set_xlabel("x [m]", fontsize=28)
#         axs.set_xticks([10, 20, 30, 40, 50])
#         axs.set_xticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
#         axs.set_yticks([10, 20, 30, 40, 50])
#         axs.set_yticklabels([0.4, 0.8, 1.2, 1.6, 2.0])
#         fig.savefig(project_path + "/plotting/CFD_diff" + ".pdf", bbox_inches="tight")


def plot(project_path, config):
    """Runs the appropriate plotting function based on the data dimension 1D or 2D

    Args:
        project_path (string): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
    """
    if config.data_dimension == 1:
        plot_1D(project_path, config)
    elif config.data_dimension == 2:
        plot_2D(project_path, config)
