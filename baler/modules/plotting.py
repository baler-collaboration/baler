from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages


# pickle_file = "./data/cfd2/cfd.pickle"

# directory = "ConvAEDes_75"
# directory = "ConvAEDes_500"

# decompressed_cfd = f"./projects/{directory}/decompressed_output/decompressed.pickle"


def loss_plot(path_to_loss_data, output_path, config):
    loss_data = pd.read_csv(path_to_loss_data)
    str_list = ["Epochs:", "Model Name:", "Reg. Param:", "lr:", "BS:"]

    val_loss = loss_data["Val Loss"]
    train_loss = loss_data["Train Loss"]
    conf_list = [
        len(train_loss),
        config["model_name"],
        config["reg_param"],
        config["lr"],
        config["batch_size"],
    ]

    plt.figure(figsize=(10, 7))
    plt.title("Loss plot")
    plt.plot(train_loss, color="orange", label="Train Loss")
    plt.plot(val_loss, color="red", label="Validation Loss")
    for i in range(len(conf_list)):
        plt.plot([], [], " ", label=str_list[i] + " " + str(conf_list[i]))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path + "_Loss_plot.pdf")
    # plt.show()


def pickle_to_df(file):
    # From pickle to df:
    with open(file, "rb") as handle:
        data = pickle.load(handle)

        return data


def get_index_to_cut(column_index, cut, array):
    indices_to_cut = np.argwhere(array[column_index] < cut).flatten()
    return indices_to_cut


def plot_1D(project_path, config):
    output_path = project_path + "training/"
    before_path = config.input_path
    after_path = project_path + "decompressed_output/decompressed.npy"

    before = np.transpose(np.load(before_path)["data"])
    after = np.transpose(np.load(after_path))
    names = np.load(config.input_path)["names"]

    index_to_cut = get_index_to_cut(3, 1e-6, before)
    before = np.delete(before, index_to_cut, axis=1)
    after = np.delete(after, index_to_cut, axis=1)

    response = np.divide(np.subtract(after, before), before) * 100
    residual = np.subtract(after, before)

    with PdfPages(project_path + "/plotting/comparison.pdf") as pdf:
        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1])

        axsLeft = subfigs[0].subplots(2, 1, sharex=True)
        ax1 = axsLeft[0]
        ax3 = axsLeft[1]
        axsRight = subfigs[1].subplots(2, 1, sharex=False)
        ax2 = axsRight[0]
        ax4 = axsRight[1]

        number_of_columns = len(names)
        for index, column in enumerate(names):
            column_name = column.split(".")[-1]
            print(f"Plotting: {column_name} ({index+1} of {number_of_columns})")
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

            # if index == 3:
            #    break


def plot_2D(project_path, config):
    data = np.load(config.data_path)

    data_decompressed = np.load(
        project_path + "/decompressed_output/decompressed.npy"
    ).reshape(50, 50)

    diff = ((data_decompressed - data) / data) * 100

    fig, axs = plt.subplots(
        1, 3, figsize=(29.7 * (1 / 2.54), 21 * (1 / 2.54)), sharey=True
    )
    axs[0].set_title("Original", fontsize=11, y=-0.2)
    im1 = axs[0].imshow(
        data, vmin=-0.01, vmax=0.07, cmap="CMRmap", interpolation="nearest"
    )
    cb2 = plt.colorbar(im1, ax=[axs[0]], location="top")

    axs[1].set_title("Decompressed", fontsize=11, y=-0.2)
    im2 = axs[1].imshow(
        data_decompressed, vmin=-0.01, vmax=0.07, cmap="CMRmap", interpolation="nearest"
    )
    cb2 = plt.colorbar(im2, ax=[axs[1]], location="top")

    axs[2].set_title("Relative Diff. [%]", fontsize=11, y=-0.2)
    im3 = axs[2].imshow(diff, vmin=-10, vmax=10, cmap="cool_r", interpolation="nearest")
    cb2 = plt.colorbar(im3, ax=[axs[2]], location="top")

    plt.ylim(0, 50)
    plt.xlim(0, 50)
    fig.suptitle(
        "Compressed file is 10% the size of original,\n75 epochs (3.5 min)",
        y=0.9,
        fontsize=16,
    )
    # fig.suptitle('Compressed file is 10% the size of original,\n500 epochs (20 min)',y=0.9, fontsize=16)

    fig.savefig(project_path + "/plotting/" + "CFD.jpg", bbox_inches="tight")


def plot(project_path, config):
    if config.data_dimension == 1:
        plot_1D(project_path, config)
    elif config.data_dimension == 2:
        plot_2D(project_path, config)
