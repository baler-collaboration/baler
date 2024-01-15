import os
import time
from math import ceil

import numpy as np

import baler_compressor.helper as helper
import gzip
import sys


def run(input_data_path, model_output_path, normalization_features, config):
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

    verbose = config.verbose

    print("Compressing...")
    start = time.time()
    normalization_features = []

    if config.apply_normalization:
        normalization_features = normalization_features

    (
        compressed,
        error_bound_batch,
        error_bound_deltas,
        error_bound_index,
        original_shape,
    ) = helper.compress(
        model_path=model_output_path,
        config=config,
    )

    end = time.time()

    print("Compression took:", f"{(end - start) / 60:.3} minutes")

    names = np.load(config.input_path)["names"]

    # if config.extra_compression:
    #     if verbose:
    #         print("Extra compression selected")
    #         print(
    #             f"Saving compressed file to {os.path.join(output_path, 'compressed_output', 'compressed.npz')}"
    #         )
    #     np.savez_compressed(
    #         os.path.join(output_path, "compressed_output", "compressed.npz"),
    #         data=compressed,
    #         names=names,
    #         normalization_features=normalization_features,
    #     )
    # else:
    #     if verbose:
    #         print("Extra compression not selected")
    #         print(
    #             f"Saving compressed file to {os.path.join(output_path, 'compressed_output', 'compressed.npz')}"
    #         )
    #     np.savez(
    #         os.path.join(output_path, "compressed_output", "compressed.npz"),
    #         data=compressed,
    #         names=names,
    #         normalization_features=normalization_features,
    #     )

    # if config.save_error_bounded_deltas:
    #     error_bound_batch_index = np.array(
    #         [error_bound_batch, error_bound_index], dtype=object
    #     )
    #     f_batch_index = gzip.GzipFile(
    #         os.path.join(
    #             output_path,
    #             "compressed_output",
    #             "compressed_batch_index_metadata.npz.gz",
    #         ),
    #         "w",
    #     )
    #     f_deltas = gzip.GzipFile(
    #         os.path.join(output_path, "compressed_output", "compressed_deltas.npz.gz"),
    #         "w",
    #     )
    #     np.save(file=f_deltas, arr=error_bound_deltas)
    #     np.save(
    #         file=f_batch_index,
    #         arr=error_bound_batch_index,
    #     )
    #     f_batch_index.close()
    #     f_deltas.close()

    return (compressed, names, original_shape)
