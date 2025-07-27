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

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from tqdm import trange


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


def plot_1D(output_path: str, config, extra_path):
    """General plotting for 1D data, for example data from a '.csv' file. This function generates a pdf
        document where each page contains the before/after performance
        of each column of the 1D data

    Args:
        output_path (path): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
        extra_path (path): The path to the directory where the decompressed data is stored. 
    """

    before_path = config.input_path
    if extra_path == "decompressed_output":
        after_path = os.path.join(output_path, extra_path, "decompressed.npz")
        write_path = os.path.join(output_path, "plotting", "comparison.pdf")
    else:
        after_path = os.path.join(extra_path, "decompressed.npz")
        write_path = os.path.join(extra_path, "comparison.pdf")

    before = np.transpose(np.load(before_path)["data"])
    after = np.transpose(np.load(after_path)["data"])
    names = np.load(config.input_path)["names"]

    index_to_cut = get_index_to_cut(3, 1e-6, before)
    before = np.delete(before, index_to_cut, axis=1)
    after = np.delete(after, index_to_cut, axis=1)

    # with np.errstate(divide='raise'):
    #     try:
    #         response = np.divide(np.subtract(after, before), before) * 100
    #     except FloatingPointError:
    #         print("caught divide by zero")
    #         response = 0    
    response = np.divide(np.subtract(after, before), before) * 100
    residual = np.subtract(after, before)

    with PdfPages(write_path) as pdf:
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
            try:
                ax1.set_xlim(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff)
            except ValueError:
                print(
                    f"Error setting xlim for {column_name}. Data may be too sparse or not well defined."
                )
                ax1.set_xlim(0, 10)
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


def plot_2D_old(project_path, config):
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

    if config.model_type == "convolutional" and config.model_name == "Conv_AE_3D":
        data_decompressed = data_decompressed.reshape(
            data_decompressed.shape[0] * data_decompressed.shape[2],
            1,
            data_decompressed.shape[3],
            data_decompressed.shape[4],
        )

    print("=== Plotting ===")
    for ind in trange(num_tiles):
        if config.model_type == "convolutional":
            tile_data_decompressed = data_decompressed[ind][0] * 0.04 * 1000
        elif config.model_type == "dense":
            tile_data_decompressed = data_decompressed[ind] * 0.04 * 1000
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


def plot_2D(project_path, config, extra_path):
    import sys

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

    if config.convert_to_blocks:
        data_decompressed = data_decompressed.reshape(
            data.shape[0], data.shape[1], data.shape[2]
        )

    if data.shape[0] > 1:
        num_tiles = data.shape[0]
    else:
        num_tiles = 1

    # if config.model_type == "convolutional" and config.model_name == "Conv_AE_3D":
    #     data_decompressed = data_decompressed.reshape(
    #         data_decompressed.shape[0] * data_decompressed.shape[2],
    #         1,
    #         data_decompressed.shape[3],
    #         data_decompressed.shape[4],
    #     )

    print("=== Plotting ===")
    for ind in trange(num_tiles):
        # if config.model_type == "convolutional":
        #     tile_data_decompressed = data_decompressed[ind][0]
        # elif config.model_type == "dense":
        #     tile_data_decompressed = data_decompressed[ind][0]
        tile_data = data[ind]
        tile_data_decompressed = data_decompressed[ind]

        diff = tile_data - tile_data_decompressed

        max_value = np.amax([np.amax(tile_data), np.amax(tile_data_decompressed)])
        min_value = np.amin([np.amin(tile_data), np.amin(tile_data_decompressed)])

        fig, axs = plt.subplots(
            1, 3, figsize=(29.7 * (1 / 2.54), 10 * (1 / 2.54)), sharey=True
        )
        axs[0].set_title("Original", fontsize=11)
        im1 = axs[0].imshow(tile_data, vmax=max_value, vmin=min_value)
        axs[1].set_title("Reconstructed", fontsize=11)
        im2 = axs[1].imshow(tile_data_decompressed, vmax=max_value, vmin=min_value)
        axs[2].set_title("Difference", fontsize=11)
        im3 = axs[2].imshow(diff, vmax=max_value, vmin=min_value)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.815, 0.2, 0.02, 0.59])
        cb2 = fig.colorbar(im3, cax=cbar_ax, location="right", aspect=10)

        fig.savefig(
            project_path + "/plotting/CFD" + str(ind) + ".png", bbox_inches="tight"
        )
        # sys.exit()


def plot(project_path, config, extra_path="decompressed_output"):
    """Runs the appropriate plotting function based on the data dimension 1D or 2D

    Args:
        extra_path (path): The path to the project directory
        config (dataclass): The config class containing attributes set in the config file
    """
    if config.data_dimension == 1:
        plot_1D(project_path, config, extra_path)
    elif config.data_dimension == 2:
        plot_2D(project_path, config, extra_path)


def plot_comparison_summary(results, output_path, original_size_mb):
    """
    Generates a single PDF page with summary plots comparing all benchmark results.

    Args:
        results (list[BenchmarkResult]): A list of BenchmarkResult data objects.
        output_path (str): The main output directory for the project.
        original_size_mb (float): The size of the original uncompressed file in MB.
    """
    print("=== Plotting Comparison Summary ===")

    # Sort results by RMSE for consistent plotting order
    sorted_results = sorted(results, key=lambda r: r.rmse)

    # Extract data into lists for easy plotting
    names = [r.name for r in sorted_results]
    rmse_values = [r.rmse for r in sorted_results]
    compress_times = [r.compress_time_sec for r in sorted_results]
    decompress_times = [r.decompress_time_sec for r in sorted_results]

    # Calculate compression ratios
    ratios = [
        original_size_mb / r.size_mb if r.size_mb > 0 else 0 for r in sorted_results
    ]

    # --- Create the plots ---
    fig, axs = plt.subplots(2, 2, figsize=(18, 24))
    fig.suptitle("Compression Benchmark Summary", fontsize=20, y=1.02)

    # 1. RMSE (Error) Plot
    ax1 = axs[0, 0]
    ax1.bar(names, rmse_values, color="skyblue")
    ax1.set_title("Reconstruction Error (RMSE)")
    ax1.set_ylabel("RMSE (lower is better)")
    ax1.set_yscale("log")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # 2. Performance (Time) Plot - Grouped Bar Chart
    ax2 = axs[0, 1]
    x = np.arange(len(names))
    width = 0.35
    ax2.bar(
        x - width / 2, compress_times, width, label="Compression Time", color="coral"
    )
    ax2.bar(
        x + width / 2,
        decompress_times,
        width,
        label="Decompression Time",
        color="lightgreen",
    )
    ax2.set_title("Performance")
    ax2.set_ylabel("Time (s, lower is better)")
    ax2.set_xticks(x, names)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # 3. Compression Ratio Plot
    ax3 = axs[1, 0]
    ax3.bar(names, ratios, color="mediumpurple")
    ax3.set_title("Compression Ratio")
    ax3.set_ylabel("Ratio (Original / Compressed, higher is better)")
    ax3.axhline(
        y=1, color="gray", linestyle="--", linewidth=1
    )  # Line for no compression
    ax3.grid(axis="y", linestyle="--", alpha=0.7)

    # 4. Trade-off Plot (Ratio vs. Error)
    ax4 = axs[1, 1]
    ax4.scatter(ratios, rmse_values, color="crimson", zorder=5)
    for i, name in enumerate(names):
        ax4.text(ratios[i] * 1.02, rmse_values[i], name, fontsize=9, rotation=45)
    ax4.set_title("Trade-off: Compression Ratio vs. Error")
    ax4.set_xlabel("Compression Ratio (higher is better)")
    ax4.set_ylabel("RMSE (lower is better)")
    ax4.set_yscale("log")
    ax4.set_xscale("log")  # Log scale for ratio can also be useful
    ax4.grid(True, which="both", ls="--", alpha=0.6)
    # Highlight the "Pareto frontier" or ideal corner
    ax4.annotate(
        "Ideal Region",
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        xytext=(0.6, 0.3),
        textcoords="axes fraction",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.3),
    )

    # Improve layout for all subplots
    for ax in axs.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=3)

    # Save the figure
    plot_dir = os.path.join(output_path, "plotting")
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "comparison_summary.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    # print(f"  -> Comparison summary plot saved to: {save_path}")


