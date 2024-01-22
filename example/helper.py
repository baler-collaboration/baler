import numpy as np
import matplotlib.pyplot as plt
import os


def plot2D(original, reconstructed):
    plt.close()
    diff = original - reconstructed

    max_value = np.amax([np.amax(original), np.amax(reconstructed)])
    min_value = np.amin([np.amin(original), np.amin(reconstructed)])

    fig, axs = plt.subplots(
        1, 3, figsize=(29.7 * (1 / 2.54), 10 * (1 / 2.54)), sharey=False
    )
    axs[0].set_title("Original", fontsize=11)
    im1 = axs[0].imshow(original, vmax=max_value, vmin=min_value)
    axs[0].invert_yaxis()

    axs[1].set_title("Reconstructed", fontsize=11)
    im2 = axs[1].imshow(reconstructed, vmax=max_value, vmin=min_value)
    axs[1].invert_yaxis()

    axs[2].set_title("Difference", fontsize=11)
    im3 = axs[2].imshow(diff, vmax=max_value, vmin=min_value)
    axs[2].invert_yaxis()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.815, 0.2, 0.02, 0.59])
    cb2 = fig.colorbar(im3, cax=cbar_ax, aspect=10)
    return fig


def loss_plot(loss_data, output_path, config):
    """This function Plots the loss from the training and saves it

    Args:
        path_to_loss_data (string): Path to file containing loss plot data generated during training
        output_path (path): Directory path to which the loss plot is saved
        config (dataclass): The config class containing attributes set in the config file
    """
    # loss_data = np.load(path_to_loss_data)
    str_list = ["Epochs:", "Model Name:", "Reg. Param:", "lr:", "BS:"]

    train_loss = loss_data[0]
    val_loss = loss_data[1]
    conf_list = [
        len(train_loss),
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
    plt.savefig(os.path.join(output_path, "Loss_plot.png"))
    # plt.show()
