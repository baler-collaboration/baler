import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import modules.data_processing as data_processing
import modules.helper as helper


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * round(y, 2))

    # The percent symbol needs escaping in latex
    if mpl.rcParams["text.usetex"] is True:
        return s + r"$\%$"
    else:
        return s + "%"


def loss_plot(path_to_loss_data, output_path, config):
    loss_data = pd.read_csv(path_to_loss_data)
    str_list = ["Epochs:", "Model Name:", "Reg. Param:", "lr:", "BS:"]

    val_loss = loss_data["Val Loss"]
    train_loss = loss_data["Train Loss"]
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
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(output_path + "_Loss_plot.pdf")
    # plt.show()


def plot(project_path):
    output_path = project_path + "plotting/"
    before_path = project_path + "training/before.pickle"
    after_path = project_path + "training/after.pickle"

    with open(before_path, "rb") as handle:
        before = pickle.load(handle)
    with open(after_path, "rb") as handle:
        after = pickle.load(handle)
    before = np.array(before)
    # Added because plotting is not supported for non-DataFrame objects yet.

    column_names = helper.from_pickle(project_path+"compressed_output/column_names.pickle")
    
    before = pd.DataFrame(before, columns=column_names)
    after = pd.DataFrame(after, columns=column_names)

    columns = data_processing.get_columns(before)
    number_of_columns = len(columns)

    with PdfPages(output_path + "comparison.pdf") as pdf:
        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1])
        
        axsLeft = subfigs[0].subplots(2, 1, sharex=True)
        ax1=axsLeft[0]
        ax3=axsLeft[1]
        axsRight = subfigs[1].subplots()
        ax2=axsRight

        response = (after - before) / before

        for index, column in enumerate(columns):
            print(f"{index} of {number_of_columns}")

            response_list = list(filter(lambda p: -20 <= p <= 20, response[column]))
            response_RMS = data_processing.RMS_function(response_norm=response_list)

            counts_before, bins_before = np.histogram(
                before[column], bins=np.linspace(0, 1,201)
            )
            hist_before = ax1.hist(
                bins_before[:-1], bins_before, weights=counts_before, label="Before"
            )
            # counts_after, bins_after = np.histogram(after[column],bins=np.arange(minimum,maximum,step))
            counts_after, bins_after = np.histogram(
                after[column], bins=np.linspace(0, 1,201)
            )
            hist_after = ax1.hist(
                bins_after[:-1],
                bins_after,
                weights=counts_after,
                label="After",
                histtype="step",
            )
            ax1.plot([], [], " ", label=f"Counts before: {len(before)}")
            ax1.plot([], [], " ", label=f"Counts after: {len(after)}")
            ax1.set_title(f"{column} Distribution")
            ax1.set_xlabel(f"{column}", ha="right", x=1.0)
            #ax1.set_xticks([])
            ax1.set_ylabel("Counts", ha="right", y=1.0)
            ax1.set_yscale("log")
            ax1.legend(loc="best")

            #ax1.set_xlim(x_min-(diff/2)*0.1,x_max+abs(x_max*0.1))
            ax1.set_xlim(-0.05,1.05)
            # Residual subplot in comparison
            #divider = make_axes_locatable(ax1)
            #ax3 = divider.append_axes("bottom", size="20%", pad=0.25)
            #ax1.figure.add_axes(ax3)
            #ax3.bar(bins_after[:-1], height=(hist_after[0] - hist_before[0])/hist_before[0])
            data_bin_centers = bins_after[:-1]+(bins_after[1:]-bins_after[:-1])/2
            ax3.scatter(data_bin_centers, (counts_after - counts_before),) # FIXME: Dividing by zero
            ax3.axhline(y=0, linewidth=0.2, color="black")
            ax3.set_ylim(-200, 200)
            #ax3.set_ylabel("(after - before)/before")
            ax3.set_ylabel("Residual")
            #ax3.set_xlim(x_min-abs(x_min*0.1),x_max+abs(x_max*0.1))

            #            minimum = min(response[column])
            #            maximum = max(response[column])
            #            diff = maximum - minimum
            #            if diff == np.inf or diff == 0:
            #                pdf.savefig()
            #                ax2.clear()
            #                ax1.clear()
            #                continue
            #            step = diff/100
            # counts_response, bins_response = np.histogram(response[column],bins=np.arange(minimum,maximum,step))
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
                label=f"Mean {round(np.mean(response_list),8)}",
            )
            ax2.plot([], [], " ", label=f"RMS: {round(response_RMS,8)}")

            # To have percent on the x-axis
            # formatter = mpl.ticker.FuncFormatter(to_percent)
            # ax2.xaxis.set_major_formatter(formatter)
            ax2.set_title(f"{column} Response")
            ax2.set_xlabel(f"{column} Response", ha="right", x=1.0)
            ax2.set_ylabel("Counts", ha="right", y=1.0)
            ax2.legend(loc="best")

            pdf.savefig()
            ax2.clear()
            ax1.clear()
            ax3.clear()

            if index==3: break
