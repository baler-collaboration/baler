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
import time

import numpy as np
import torch

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
    config, mode, workspace_name, project_name = helper.get_arguments()
    project_path = os.path.join("workspaces", workspace_name, project_name)
    output_path = os.path.join(project_path, "output")

    if mode == "newProject":
        helper.create_new_project(workspace_name, project_name)
    elif mode == "train":
        perform_training(output_path, config)
    elif mode == "diagnose":
        perform_diagnostics(output_path)
    elif mode == "compress":
        perform_compression(output_path, config)
    elif mode == "decompress":
        perform_decompression(output_path, config)
    elif mode == "plot":
        perform_plotting(output_path, config)
    elif mode == "info":
        print_info(output_path, config)
    else:
        raise NameError(
            "Baler mode "
            + mode
            + " not recognised. Use baler --help to see available modes."
        )


def perform_training(output_path, config):
    """Main function calling the training functions, ran when --mode=train is selected.
        The three main functions this calls are: `helper.process`, `helper.mode_init` and `helper.training`.

        Depending on `config.data_dimensions`, the calculated latent space size will differ.

    Args:
        output_path (path): Selects base path for determining output path
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

    training_path = os.path.join(output_path, "training")
    print("path:", training_path)

    trained_model = helper.train(
        model, number_of_columns, train_set_norm, test_set_norm, training_path, config
    )

    if config.apply_normalization:
        np.save(
            os.path.join(training_path, "normalization_features.npy"),
            normalization_features,
        )
    helper.model_saver(
        trained_model, os.path.join(output_path, "compressed_output", "model.pt")
    )


def perform_diagnostics(project_path):
    print("Performing diagnostics...")
    output_path = os.path.join(project_path, "plotting")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = os.path.join(project_path, "training", "activations.npy")
    helper.diagnose(input_path, output_path)


def perform_plotting(output_path, config):
    """Main function calling the two plotting functions, ran when --mode=plot is selected.
       The two main functions this calls are: `helper.plotter` and `helper.loss_plotter`

    Args:
        output_path (string): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
    """
    helper.loss_plotter(
        os.path.join(output_path, "training", "loss_data.npy"), output_path, config
    )
    helper.plotter(output_path, config)


def perform_compression(output_path, config):
    """Main function calling the compression functions, ran when --mode=compress is selected.
       The main function being called here is: `helper.compress`

        If `config.extra_compression` is selected, the compressed file is further compressed via zip
        Else, the function returns a compressed file of `.npz`, only compressed by Baler.

    Args:
        output_path (path): Selects base path for determining output path
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
            os.path.join(output_path, "training", "normalization_features.npy")
        )

    compressed = helper.compress(
        model_path=os.path.join(output_path, "compressed_output", "model.pt"),
        config=config,
    )

    end = time.time()

    print("Compression took:", f"{(end - start) / 60:.3} minutes")

    names = np.load(config.input_path)["names"]

    if config.extra_compression:
        np.savez_compressed(
            os.path.join(output_path, "compressed_output", "compressed.npz"),
            data=compressed,
            names=names,
            normalization_features=normalization_features,
        )
    else:
        np.savez(
            os.path.join(output_path, "compressed_output", "compressed.npz"),
            data=compressed,
            names=names,
            normalization_features=normalization_features,
        )


def perform_decompression(output_path, config):
    """Main function calling the decompression functions, ran when --mode=decompress is selected.
       The main function being called here is: `helper.decompress`

        If `config.apply_normalization=True` the output is un-normalized with the same normalization features saved from `perform_training()`.

    Args:
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
    """
    print("Decompressing...")

    start = time.time()
    model_name = config.model_name
    decompressed, names, normalization_features = helper.decompress(
        model_path=os.path.join(output_path, "compressed_output", "model.pt"),
        input_path=os.path.join(output_path, "compressed_output", "compressed.npz"),
        model_name=model_name,
        config=config,
    )

    if config.apply_normalization:
        print("Un-normalizing...")
        normalization_features = np.load(
            os.path.join(output_path, "training", "normalization_features.npy"),
        )

        decompressed = helper.renormalize(
            decompressed,
            normalization_features[0],
            normalization_features[1],
        )

    try:
        type_list = config.type_list
        decompressed = np.transpose(decompressed)
        for index, column in enumerate(decompressed):
            decompressed[index] = decompressed[index].astype(type_list[index])
        decompressed = np.transpose(decompressed)
    except AttributeError:
        pass

    end = time.time()
    print("Decompression took:", f"{(end - start) / 60:.3} minutes")

    if config.extra_compression:
        np.savez_compressed(
            os.path.join(output_path, "decompressed_output", "decompressed.npz"),
            data=decompressed,
            names=names,
        )
    else:
        decompressed = decompressed.detach().numpy()
        np.savez(
            os.path.join(output_path, "decompressed_output", "decompressed.npz"),
            data=decompressed,
            names=names,
        )


def print_info(output_path, config):
    """Function which prints information about your total compression ratios and the file sizes.

    Args:meta_data
        output_path (string): Selects path to project from which one wants to obtain file information
        config (dataClass): Base class selecting user inputs
    """
    print(
        "================================== \n Information about your compression \n================================== "
    )

    original = config.input_path
    compressed_path = os.path.join(output_path, "compressed_output")
    decompressed_path = os.path.join(output_path, "decompressed_output")
    training_path = os.path.join(output_path, "training")

    model = os.path.join(compressed_path, "model.pt")
    compressed = os.path.join(compressed_path, "compressed.npz")
    decompressed = os.path.join(decompressed_path, "decompressed.npz")

    meta_data = [
        model,
        os.path.join(training_path, "loss_data.npy"),
        os.path.join(training_path, "normalization_features.npy"),
    ]

    meta_data_stats = [
        os.stat(meta_data[file]).st_size / (1024 * 1024)
        for file in range(len(meta_data))
    ]

    files = [original, compressed, decompressed]
    file_stats = [
        os.stat(files[file]).st_size / (1024 * 1024) for file in range(len(files))
    ]

    print(
        f"\nCompressed file is {round(file_stats[1] / file_stats[0], 4) * 100}% the size of the original\n"
    )
    print(f"File size before compression: {round(file_stats[0], 4)} MB\n")
    print(f"Compressed file size: {round(file_stats[1], 4)} MB\n")
    print(f"De-compressed file size: {round(file_stats[2], 4)} MB\n")
    print(f"Compression ratio: {round(file_stats[0] / file_stats[1], 4)}\n")
    print(
        f"The meta-data saved has a total size of: {round(sum(meta_data_stats),4)} MB\n"
    )
    print(
        f"Combined, the actual compression ratio is: {round((file_stats[0])/(file_stats[1] + sum(meta_data_stats)),4)}"
    )
    print("\n ==================================")

    ## TODO: Add way to print how much your data has been distorted


if __name__ == "__main__":
    main()
