from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


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


def plot(project_path, config):

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
