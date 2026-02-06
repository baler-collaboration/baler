# Copyright 2022-2025 Baler Contributors

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
from math import ceil
import numpy as np

from .modules import helper
from .modules import compare
import gzip
# from .modules.profiling import pytorch_profile
import blosc2

__all__ = (
    "perform_compression",
    "perform_decompression",
    "perform_diagnostics",
    "perform_plotting",
    "perform_training",
    "print_info",
    "perform_comparison",
)


def main():
    """Calls different functions depending on argument parsed in command line.

        - if --mode=newProject: call `helper.create_new_project` and create a new project sub directory with config file
        - if --mode=train: call `perform_training` and train the network on given data and based on the config file and check if profilers are enabled
        - if --mode=diagnose: call `perform_diagnostics` and diagnose the training process by plotting the activations of the layers.
        - if --mode=compress: call `perform_compression` and compress the given data using the model trained in `--mode=train`
        - if --mode=decompress: call `perform_decompression` and decompress the compressed file outputted from `--mode=compress`
        - if --mode=plot: call `perform_plotting` and plot the comparison between the original data and the decompressed data from `--mode=decompress`. Also plots the loss plot from the trained network.
        - if --mode=info: call `print_info` and print information about the compression ratio and file sizes.
        - if --mode=convert_with_hls4ml: call `helper.perform_hls4ml_conversion` and create an hls4ml project containing the converted model.
        - if --mode=compare: call `perform_comparison` and compare the compressed data with non-AE lossy compression algorithms.


    Raises:
        NameError: Raises error if the chosen mode does not exist.
    """
    (
        config,
        mode,
        workspace_name,
        project_name,
        verbose,
    ) = helper.get_arguments()
    project_path = os.path.join("workspaces", workspace_name, project_name)
    output_path = os.path.join(project_path, "output")

    if mode == "newProject":
        helper.create_new_project(workspace_name, project_name, verbose)
    elif mode == "train":
        perform_training(output_path, config, project_name, verbose)
    elif mode == "diagnose":
        perform_diagnostics(output_path, verbose)
    elif mode == "compress":
        perform_compression(output_path, config, verbose)
    elif mode == "decompress":
        perform_decompression(output_path, config, verbose)
    elif mode == "plot":
        perform_plotting(output_path, config, verbose)
    elif mode == "info":
        print_info(output_path, config)
    elif mode == "convert_with_hls4ml":
        helper.perform_hls4ml_conversion(output_path, config)
    elif mode == "compare":
        perform_comparison(output_path, config, project_name, verbose)
    else:
        raise NameError(
            "Baler mode "
            + mode
            + " not recognised. Use baler --help to see available modes."
        )


def perform_training(output_path, config, project_name, verbose: bool):
    """Main function calling the training functions, ran when --mode=train is selected.
        The three functions called are: `helper.process`, `helper.mode_init` and `helper.training`.

        Depending on `config.data_dimensions`, the calculated latent space size will differ.

    Args:
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information

    Raises:
        NameError: Baler currently only supports 1D (e.g. HEP) or 2D (e.g. CFD) data as inputs.
    """
    green_code_timer_start = time.perf_counter()
    (
        train_set_norm,
        test_set_norm,
        normalization_features,
        original_shape,
    ) = helper.process(
        config.input_path,
        config.custom_norm,
        config.test_size,
        config.apply_normalization,
        config.convert_to_blocks if hasattr(config, "convert_to_blocks") else None,
        verbose,
    )

    if verbose:
        print("Training and testing sets normalized")

    try:
        n_features = 0
        if config.data_dimension == 1:
            number_of_columns = train_set_norm.shape[1]
            config.latent_space_size = ceil(
                number_of_columns / config.compression_ratio
            )
            config.number_of_columns = number_of_columns
            n_features = number_of_columns
        elif config.data_dimension == 2:
            if config.model_type == "dense":
                number_of_rows = train_set_norm.shape[1]
                number_of_columns = train_set_norm.shape[2]
                n_features = number_of_columns * number_of_rows
            else:
                number_of_rows = original_shape[1]
                number_of_columns = original_shape[2]
                n_features = number_of_columns
            config.latent_space_size = ceil(
                (number_of_rows * number_of_columns) / config.compression_ratio
            )
            config.number_of_columns = number_of_columns
        else:
            raise NameError(
                "Data dimension can only be 1 or 2. Got config.data_dimension value = "
                + str(config.data_dimension)
            )
    except AttributeError:
        if verbose:
            print(
                f"{config.number_of_columns} -> {config.latent_space_size} dimensions"
            )
        assert number_of_columns == config.number_of_columns

    if verbose:
        print(
            f"Intitalizing Model with Latent Size - {config.latent_space_size} and Features - {n_features}"
        )

    device = helper.get_device()
    if verbose:
        print(f"Device used for training: {device}")

    model_object = helper.model_init(config.model_name)
    model = model_object(n_features=n_features, z_dim=config.latent_space_size)
    model.to(device)

    if config.model_name == "Conv_AE_3D" and hasattr(
        config, "compress_to_latent_space"
    ):
        model.set_compress_to_latent_space(config.compress_to_latent_space)

    if verbose:
        print(f"Model architecture:\n{model}")

    training_path = os.path.join(output_path, "training")
    if verbose:
        print(f"Training path: {training_path}")

    trained_model = helper.train(
        model, number_of_columns, train_set_norm, test_set_norm, training_path, config
    )

    if verbose:
        print("Training complete")

    if config.apply_normalization:
        np.save(
            os.path.join(training_path, "normalization_features.npy"),
            normalization_features,
        )
        if verbose:
            print(
                f"Normalization features saved to {os.path.join(training_path, 'normalization_features.npy')}"
            )

    if config.separate_model_saving:
        helper.encoder_decoder_saver(
            trained_model,
            os.path.join(output_path, "compressed_output", "encoder.pt"),
            os.path.join(output_path, "compressed_output", "decoder.pt"),
        )
    else:
        helper.model_saver(
            trained_model, os.path.join(output_path, "compressed_output", "model.pt")
        )
    if verbose:
        print(
            f"Model saved to {os.path.join(output_path, 'compressed_output', 'model.pt')}"
        )

        print("\nThe model has the following structure:")
        print(model.type)

    green_code_timer_end = time.perf_counter()
    helper.green_code_tracking(
        start=green_code_timer_start,
        end=green_code_timer_end,
        title=f"{project_name} - Model Training",
    )


def perform_diagnostics(project_path, verbose: bool):
    output_path = os.path.join(project_path, "plotting")
    if verbose:
        print("Performing diagnostics")
        print(f"Saving plots to {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = os.path.join(project_path, "training", "activations.npy")
    helper.diagnose(input_path, output_path)


def perform_plotting(output_path, config, verbose: bool):
    """Main function calling the two plotting functions, ran when --mode=plot is selected.
       The two main functions this calls are: `helper.plotter` and `helper.loss_plotter`

    Args:
        output_path (string): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """
    if verbose:
        print("Plotting...")
        print(f"Saving plots to {output_path}")
    helper.loss_plotter(
        os.path.join(output_path, "training", "loss_data.npy"), output_path, config
    )
    helper.plotter(output_path, config)


def perform_compression(output_path, config, verbose: bool):
    """Main function calling the compression functions, ran when --mode=compress is selected.
       The main function being called here is: `helper.compress`

        If `config.extra_compression` is selected, the compressed file is further compressed via zip
        Else, the function returns a compressed file of `.npz`, only compressed by Baler.

    Args:
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information

    Outputs:
        An `.npz` file which includes:
        - The compressed data
        - The data headers
        - Normalization features if `config.apply_normalization=True`
    """
    print("Compressing...")
    start = time.perf_counter()
    normalization_features = []

    if config.apply_normalization:
        normalization_features = np.load(
            os.path.join(output_path, "training", "normalization_features.npy")
        )
    if config.separate_model_saving:
        (
            compressed,
            error_bound_batch,
            error_bound_deltas,
            error_bound_index,
        ) = helper.compress(
            model_path=os.path.join(output_path, "compressed_output", "encoder.pt"),
            config=config,
        )
    else:
        (
            compressed,
            error_bound_batch,
            error_bound_deltas,
            error_bound_index,
        ) = helper.compress(
            model_path=os.path.join(output_path, "compressed_output", "model.pt"),
            config=config,
        )

    end = time.perf_counter()

    print("Compression took:", f"{(end - start):.4f} seconds")

    names = np.load(config.input_path)["names"]

    if config.extra_compression:
        if verbose:
            print("Extra compression selected")
            print(
                f"Saving compressed file to {os.path.join(output_path, 'compressed_output', 'compressed.npz')}"
            )
        np.savez_compressed(
            os.path.join(output_path, "compressed_output", "compressed.npz"),
            data=compressed,
            names=names,
            normalization_features=normalization_features,
        )
    else:
        if verbose:
            print("Extra compression not selected")
            print(
                f"Saving compressed file to {os.path.join(output_path, 'compressed_output', 'compressed.npz')}"
            )
        np.savez(
            os.path.join(output_path, "compressed_output", "compressed.npz"),
            data=compressed,
            names=names,
            normalization_features=normalization_features,
        )
    if config.save_error_bounded_deltas:
        error_bound_batch_index = np.array(
            [error_bound_batch, error_bound_index], dtype=object
        )
        f_batch_index = gzip.GzipFile(
            os.path.join(
                output_path,
                "compressed_output",
                "compressed_batch_index_metadata.npz.gz",
            ),
            "w",
        )
        f_deltas = gzip.GzipFile(
            os.path.join(output_path, "compressed_output", "compressed_deltas.npz.gz"),
            "w",
        )
        np.save(file=f_deltas, arr=error_bound_deltas)
        np.save(
            file=f_batch_index,
            arr=error_bound_batch_index,
        )
        f_batch_index.close()
        f_deltas.close()


def perform_decompression(output_path, config, verbose: bool):
    """Main function calling the decompression functions, ran when --mode=decompress is selected.
       The main function being called here is: `helper.decompress`

        If `config.apply_normalization=True` the output is un-normalized with the same normalization features saved from `perform_training()`.

    Args:
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """
    print("Decompressing...")

    start = time.perf_counter()
    model_name = config.model_name
    data_before = np.load(config.input_path)["data"]
    if config.separate_model_saving:
        decompressed, names, normalization_features = helper.decompress(
            model_path=os.path.join(output_path, "compressed_output", "decoder.pt"),
            input_path=os.path.join(output_path, "compressed_output", "compressed.npz"),
            input_path_deltas=os.path.join(
                output_path, "compressed_output", "compressed_deltas.npz.gz"
            ),
            input_batch_index=os.path.join(
                output_path,
                "compressed_output",
                "compressed_batch_index_metadata.npz.gz",
            ),
            model_name=model_name,
            config=config,
            output_path=output_path,
            original_shape=data_before.shape,
        )
    else:
        decompressed, names, normalization_features = helper.decompress(
            model_path=os.path.join(output_path, "compressed_output", "model.pt"),
            input_path=os.path.join(output_path, "compressed_output", "compressed.npz"),
            input_path_deltas=os.path.join(
                output_path, "compressed_output", "compressed_deltas.npz.gz"
            ),
            input_batch_index=os.path.join(
                output_path,
                "compressed_output",
                "compressed_batch_index_metadata.npz.gz",
            ),
            model_name=model_name,
            config=config,
            output_path=output_path,
            original_shape=data_before.shape,
        )
    if verbose:
        print(f"Model used: {model_name}")

    if hasattr(config, "convert_to_blocks") and config.convert_to_blocks:
        print(
            "Converting Blocked Data into Standard Format. Old Shape - ",
            decompressed.shape,
            "Target Shape - ",
            data_before.shape,
        )
        if config.model_type == "dense":
            decompressed = decompressed.reshape(
                data_before.shape[0], data_before.shape[1], data_before.shape[2]
            )
        else:
            decompressed = decompressed.reshape(
                data_before.shape[0], 1, data_before.shape[1], data_before.shape[2]
            )

    if config.apply_normalization:
        print("Un-normalizing...")
        normalization_features = np.load(
            os.path.join(output_path, "training", "normalization_features.npy"),
        )
        if verbose:
            print(
                f"Normalization features loaded from {os.path.join(output_path, 'training', 'normalization_features.npy')}"
            )

        decompressed = helper.renormalize(
            decompressed,
            normalization_features[0],
            normalization_features[1],
        )

    try:
        if verbose:
            print("Converting to original data types")
        type_list = config.type_list
        decompressed = np.transpose(decompressed)
        for index, column in enumerate(decompressed):
            decompressed[index] = decompressed[index].astype(type_list[index])
        decompressed = np.transpose(decompressed)
    except AttributeError:
        pass

    end = time.perf_counter()
    print("Decompression took:", f"{(end - start):.4f} seconds")

    if config.extra_compression:
        if verbose:
            print("Extra compression selected")
            print(
                f"Saving decompressed file to {os.path.join(output_path, 'decompressed_output', 'decompressed.npz')}"
            )
        np.savez_compressed(
            os.path.join(output_path, "decompressed_output", "decompressed.npz"),
            data=decompressed,
            names=names,
        )
    else:
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
    ]

    loss_path = os.path.join(training_path, "loss_data.npy")
    if os.path.exists(loss_path):
        meta_data.append(os.path.join(training_path, "loss_data.npy"))

    norm_path = os.path.join(training_path, "normalization_features.npy")
    if os.path.exists(norm_path):
        meta_data.append(os.path.join(training_path, "normalization_features.npy"))

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


def perform_comparison(output_path, config, project_name, verbose):
    """
    Runs a series of compression benchmarks and prints a summary table.
    This function orchestrates a comparison between the project's Baler model
    and other standard compression algorithms like Downcasting, ZFP, Blosc2,
    and SZ3. It loads the original data, sets up each benchmark with specific
    configurations, and then executes them sequentially.
    After running all benchmarks, it collects performance metrics such as
    compression ratio, reconstruction error (RMSE, Max Error), PSNR, and
    compression/decompression times. Finally, it sorts the results by RMSE
    and prints a formatted summary table to the console for easy comparison.
    Args:
        output_path (str): The base directory path where benchmark outputs,
            including compressed files, will be stored. Each benchmark may
            create its own subdirectory within this path.
        config (object): A configuration object containing project settings,
            such as the input data path (`config.input_path`) and the Baler
            model name (`config.model_name`).
        verbose (bool): A flag passed to the compression and decompression
            functions to control the verbosity of their output.
    """

    green_code_timer_start = time.perf_counter()

    original_path = config.input_path
    original_npz = np.load(original_path)
    data_original = original_npz["data"]
    names_original = original_npz["names"]

    # Calculate original file size for compression ratio calculation
    try:
        original_size_bytes = os.path.getsize(original_path)
        original_size_mb = original_size_bytes / (1024 * 1024)
        print(f"Original input file size: {original_size_mb:.3f} MB ({original_path})")
    except FileNotFoundError:
        print(
            f"Warning: Could not find original file at {original_path} to calculate compression ratio."
        )
        original_size_mb = 0

    # Define all the benchmarks we want to run
    benchmarks_to_run = []

    # 1. Baler
    benchmarks_to_run.append(
        compare.BalerBenchmark(
            name=f"Baler ({config.model_name})",
            output_path=output_path,  # Baler uses the main project output path
            compress_func=lambda: perform_compression(output_path, config, verbose),
            decompress_func=lambda: perform_decompression(output_path, config, verbose),
            data_original=data_original,
            names_original=names_original,
        )
    )


    # 2. Downcast float32 - only valid when using float64 inputs
    if config.float_dtype == "float32":
        pass
    else:
        benchmarks_to_run.append(
            compare.DowncastBenchmark(
                output_dir=os.path.join(output_path, "downcast_float32"),
                data_original=data_original,
                names_original=names_original,
                target_dtype=np.float32,
            )
        )

    # 3a. ZFP using 'precision' mode (the original test)
    # The 'precision' parameter specifies the number of uncompressed bits to keep.
    # A higher precision means lower compression but better quality.
    zfp_precision = 22
    benchmarks_to_run.append(
        compare.ZFPBenchmark(
            output_dir=os.path.join(output_path, f"zfp_prec{zfp_precision}"),
            data_original=data_original,
            names_original=names_original,
            zfp_params={"precision": zfp_precision},
        )
    )

    # 3b. ZFP using 'rate' mode
    # The 'rate' parameter specifies a fixed size budget in bits per value.
    # A smaller rate means higher compression.
    zfp_rate = 8.0
    benchmarks_to_run.append(
        compare.ZFPBenchmark(
            output_dir=os.path.join(output_path, f"zfp_rate{zfp_rate}"),
            data_original=data_original,
            names_original=names_original,
            zfp_params={"rate": zfp_rate},
        )
    )

    # 3c. ZFP using 'tolerance' mode
    # The 'tolerance' parameter specifies the maximum allowed error in the compressed data.
    # A smaller tolerance means higher compression but potentially more error.
    zfp_tolerance = 1e-3
    benchmarks_to_run.append(
        compare.ZFPBenchmark(
            output_dir=os.path.join(output_path, f"zfp_tol{zfp_tolerance}"),
            data_original=data_original,
            names_original=names_original,
            zfp_params={"tolerance": zfp_tolerance},
        )
    )

    # 4a. Blosc2 with LZ4 (Lossy, Fast)
    blosc_trunc_prec = 18
    benchmarks_to_run.append(
        compare.BloscBenchmark(
            name=f"Blosc2-LZ4(prec={blosc_trunc_prec})",
            output_dir=os.path.join(output_path, f"blosc2_lz4_prec{blosc_trunc_prec}"),
            data_original=data_original,
            names_original=names_original,
            cparams={
                "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE],
                "filters_meta": [blosc_trunc_prec, 0],
                "codec": blosc2.Codec.LZ4,
                "clevel": 5,
            },
        )
    )

    # 4b. Blosc2 with ZSTD (Lossy, Aggressive Compression)
    blosc_zstd_clevel = 9
    benchmarks_to_run.append(
        compare.BloscBenchmark(
            name=f"Blosc2-ZSTD(L{blosc_zstd_clevel})",
            output_dir=os.path.join(
                output_path, f"blosc2_zstd_lossy_l{blosc_zstd_clevel}"
            ),
            data_original=data_original,
            names_original=names_original,
            cparams={
                "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
                "filters_meta": [blosc_trunc_prec, 0],
                "codec": blosc2.Codec.ZSTD,
                "clevel": blosc_zstd_clevel,
            },
        )
    )

    # TODO Implement SZ3 Benchmark

    # --- Run all benchmarks and collect results ---
    all_results = []
    for benchmark in benchmarks_to_run:
        result = benchmark.run()
        all_results.append(result)

    compare.output_benchmark_results(
        original_size_mb, all_results, project_name, output_path, verbose=verbose
    )

    if all_results:
        ## helper.plot_comparison_summary(all_results, output_path, original_size_mb)
        helper.plot_all_results(output_path, config)

    green_code_timer_end = time.perf_counter()

    with open("compression_comparison_results.txt", "a") as f:
        f.write(
            f"Total time taken: {green_code_timer_end - green_code_timer_start:.3f} seconds\n"
        )

    helper.green_code_tracking(
        start=green_code_timer_start,
        end=green_code_timer_end,
        title=f"{project_name} - Compression Comparison",
        verbose=verbose,
    )
