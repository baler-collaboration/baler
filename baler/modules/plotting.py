import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import modules.data_processing as data_processing
import modules.helper as helper

import sys


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * round(y, 2))

    # The percent symbol needs escaping in latex
    if mpl.rcParams["text.usetex"]:
        return s + r"$\%$"
    else:
        return s + "%"


def loss_plot(path_to_loss_data, output_path, config):
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
    plt.plot(val_loss, color="red", label="Validation Loss")
    for i in range(len(conf_list)):
        plt.plot([], [], " ", label=str_list[i] + " " + str(conf_list[i]))
    plt.xlabel("Epochs")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path + "_Loss_plot.pdf")
    # plt.show()


def plot(project_path, config):

    output_path = project_path + "training/"
    names_path = config.names_path
    before_path = output_path + "before.npy"
    after_path = output_path + "after.npy"

    column_names = np.load(names_path)

    before = pd.DataFrame(np.load(before_path), columns=column_names)
    after = pd.DataFrame(np.load(after_path), columns=column_names)

    response = (after - before) / before

    with PdfPages(project_path + "/plotting/comparison.pdf") as pdf:
        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1])

        axsLeft = subfigs[0].subplots(2, 1, sharex=True)
        ax1 = axsLeft[0]
        ax3 = axsLeft[1]
        axsRight = subfigs[1].subplots()
        ax2 = axsRight

        columns = list(before.columns)
        number_of_columns = len(columns)
        for index, column in enumerate(columns):
            column_name = column.split(".")[-1]
            print(f"Plotting: {column_name} ({index+1} of {number_of_columns})")
            response_list = list(filter(lambda p: -20 <= p <= 20, response[column]))
            square = np.square(response_list)
            MS = square.mean()
            response_RMS = np.sqrt(MS)

            x_min = min(before[column] + after[column])
            x_max = max(before[column] + after[column])
            x_diff = abs(x_max - x_min)

            # Before Histogram
            counts_before, bins_before = np.histogram(
                before[column],
                bins=np.linspace(x_min - 0.1 * x_diff, x_max + 0.1 * x_diff, 200),
            )
            ax1.hist(
                bins_before[:-1], bins_before, weights=counts_before, label="Before"
            )

            # After Histogram
            counts_after, bins_after = np.histogram(
                after[column],
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

            data_bin_centers = bins_after[:-1] + (bins_after[1:] - bins_after[:-1]) / 2
            ax3.scatter(
                data_bin_centers, (counts_after - counts_before), marker="."
            )  # FIXME: Dividing by zero
            ax3.axhline(y=0, linewidth=0.2, color="black")
            ax3.set_xlabel(f"{column_name}", ha="right", x=1.0)
            ax3.set_ylim(-200, 200)
            ax3.set_ylabel("Residual")

            # Response Histogram
            counts_response, bins_response = np.histogram(
                response[column], bins=np.arange(-2, 2, 0.01)
            )
            ax2.hist(
                bins_response[:-1],
                bins_response,
                weights=counts_response,
                label="Response",
            )
            ax2.axvline(
                np.mean(response_list),
                color="k",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean {round(np.mean(response_list),4)}",
            )
            ax2.plot([], [], " ", label=f"RMS: {round(response_RMS,8)}")

            ax2.set_xlabel(f"{column_name} Response", ha="right", x=1.0)
            ax2.set_ylabel("Counts", ha="right", y=1.0)
            ax2.legend(loc="best", title=f"RMS: {round(response_RMS,4)}")

            pdf.savefig()
            ax2.clear()
            ax1.clear()
            ax3.clear()

            # if index == 3:
            #    break
