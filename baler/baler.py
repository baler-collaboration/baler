import os
import time

import numpy as np

import modules.helper as helper


def main():
    """Calls different functions depending on argument parsed in command line.

        - if --mode=newProject: call `helper.create_new_project` and create a new project sub directory with config file
        - if --mode=train: call `perform_training` and train the network on given data and based on the config file
        - if --mode=compress: call `perform_compression` and compress the given data using the model trained in `--mode=train`
        - if --mode=decompress: call `perform_decompression` and decompress the compressed file outputted from `--mode=compress`
        - if --mode=plot: call `perform_plotting` and plot the comparison between the original data and the decompressed data from `--mode=decompress`. Also plots the loss plot from the trained network.


    Raises:
        NameError: Raises error if the chosen mode does not exist.
    """
    config, mode, project_name = helper.get_arguments()
    project_path = f"projects/{project_name}/"
    if mode == "newProject":
        helper.create_new_project(project_name)
    elif mode == "train":
        perform_training(project_path, config)
    elif mode == "compress":
        perform_compression(project_path, config)
    elif mode == "decompress":
        perform_decompression(config.model_name, project_path, config)
    elif mode == "plot":
        perform_plotting(project_path, config)
    elif mode == "info":
        print_info(project_path, config)
    else:
        raise NameError(
            "Baler mode "
            + mode
            + " not recognised. Use baler --help to see available modes."
        )


def perform_training(project_path, config):
    """Main function calling the training functions, ran when --mode=train is selected.
        The three main functions this calls are: `helper.process`, `helper.mode_init` and `helper.training`.

        Depending on `config.data_dimensions`, the calculated latent space size will differ.

    Args:
        project_path (string): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs

    Raises:
        NameError: Baler currently only supports 1D (e.g. HEP) or 2D (e.g. CFD) data as inputs.
    """
    train_set_norm, test_set_norm, normalization_features = helper.process(
        config.input_path,
        config.custom_norm,
        config.test_size,
        config.apply_normalization,
    )

    try:
        if config.data_dimension == 1:
            number_of_columns = train_set_norm.shape[1]
            config.latent_space_size = int(
                number_of_columns // config.compression_ratio
            )
            config.number_of_columns = number_of_columns
        elif config.data_dimension == 2:
            number_of_rows = train_set_norm.shape[1]
            number_of_columns = train_set_norm.shape[2]
            config.latent_space_size = int(
                (number_of_rows * number_of_columns) // config.compression_ratio
            )
            config.number_of_columns = number_of_columns
        else:
            raise NameError(
                "Data dimension can only be 1 or 2. Got config.data_dimension value = "
                + str(config.data_dimension)
            )
    except AttributeError:
        print(f"{config.number_of_columns} -> {config.latent_space_size} dimensions")
        assert number_of_columns == config.number_of_columns

    device = helper.get_device()

    model_object = helper.model_init(config.model_name)
    model = model_object(n_features=number_of_columns, z_dim=config.latent_space_size)
    model.to(device)

    output_path = project_path + "training/"
    trained_model = helper.train(
        model, number_of_columns, train_set_norm, test_set_norm, output_path, config
    )

    if config.apply_normalization:
        np.save(
            project_path + "training/normalization_features.npy",
            normalization_features,
        )
    helper.model_saver(trained_model, project_path + "compressed_output/model.pt")


def perform_plotting(project_path, config):
    """Main function calling the two plotting functions, ran when --mode=plot is selected.
       The two main functions this calls are: `helper.plotter` and `helper.loss_plotter`

    Args:
        project_path (string): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
    """
    output_path = project_path + "plotting/"
    helper.loss_plotter(project_path + "training/loss_data.npy", output_path, config)
    helper.plotter(project_path, config)


def perform_compression(project_path, config):
    """Main function calling the compression functions, ran when --mode=compress is selected.
       The main function being called here is: `helper.compress`

        If `config.extra_compression` is selected, the compressed file is further compressed via zip
        Else, the function returns a compressed file of `.npz`, only compressed by Baler.

    Args:
        project_path (string): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs

    Outputs:
        An `.npz` file which includes:
        - The compressed data
        - The data headers
        - Normalization features if `config.apply_normalization=True`
    """
    print("Compressing...")
    start = time.time()
    normalization_features = []

    if config.apply_normalization:
        normalization_features = np.load(
            project_path + "training/normalization_features.npy"
        )

    compressed = helper.compress(
        model_path=project_path + "compressed_output/model.pt",
        config=config,
    )
    # Converting back to numpyarray
    compressed = helper.detacher(compressed)
    end = time.time()

    print("Compression took:", f"{(end - start) / 60:.3} minutes")

    names = np.load(config.input_path)["names"]

    if config.extra_compression:
        np.savez_compressed(
            project_path + "compressed_output/compressed.npz",
            data=compressed,
            names=names,
            normalization_features=normalization_features,
        )
    else:
        np.savez(
            project_path + "compressed_output/compressed.npz",
            data=compressed,
            names=names,
            normalization_features=normalization_features,
        )


def perform_decompression(model_name, project_path, config):
    """Main function calling the decompression functions, ran when --mode=decompress is selected.
       The main function being called here is: `helper.decompress`

        If `config.apply_normalization=True` the output is un-normalized with the same normalization features saved from `perform_training()`.

    Args:
        model_name (string): Name of the model you want to use for decompression
        project_path (string): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
    """
    print("Decompressing...")

    start = time.time()
    decompressed, names, normalization_features = helper.decompress(
        model_path=project_path + "compressed_output/model.pt",
        input_path=project_path + "compressed_output/compressed.npz",
        model_name=model_name,
    )

    # Converting back to numpyarray
    decompressed = helper.detacher(decompressed)

    if config.apply_normalization:
        print("Un-normalizing...")
        normalization_features = np.load(
            project_path + "training/normalization_features.npy"
        )
        decompressed = helper.renormalize(
            decompressed,
            normalization_features[0],
            normalization_features[1],
        )
    end = time.time()
    print("Decompression took:", f"{(end - start) / 60:.3} minutes")

    if config.extra_compression:
        np.savez_compressed(
            project_path + "decompressed_output/decompressed.npz",
            data=decompressed,
            names=names,
        )
    else:
        np.savez(
            project_path + "decompressed_output/decompressed.npz",
            data=decompressed,
            names=names,
        )


def print_info(project_path, config):
    print(
        "================================== \n Information about your compression \n================================== "
    )

    pre_compression = config.input_path
    model = project_path + "compressed_output/model.pt"
    compressed = project_path + "compressed_output/compressed.npz"
    decompressed = project_path + "decompressed_output/decompressed.npz"

    files = [pre_compression, compressed, decompressed]
    q = []
    for i in range(len(files)):
        q.append(os.stat(files[i]).st_size / (1024 * 1024))

    print(
        f"\nCompressed file is {round(q[1] / q[0], 2) * 100}% the size of the original\n"
    )
    print(f"File size before compression: {round(q[0], 2)} MB")
    print(f"Compressed file size: {round(q[1], 2)} MB")
    print(f"De-compressed file size: {round(q[2], 2)} MB")
    print(f"Compression ratio: {round(q[0] / q[1], 2)}")


if __name__ == "__main__":
    main()
