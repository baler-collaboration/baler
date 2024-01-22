import os
import time
from math import ceil

import numpy as np

import baler_compressor.helper as helper
import gzip
import sys


def run(model_path, compressed_output_path, config):
    """Main function calling the decompression functions, ran when --mode=decompress is selected.
    The main function being called here is: `helper.decompress`

        If `config.apply_normalization=True` the output is un-normalized with the same normalization features saved from `perform_training()`.

    Args:
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """
    print("Decompressing...")

    output_path = compressed_output_path
    verbose = config.verbose

    start = time.time()
    # model_name = config.model_name
    # data_before = np.load(config.input_path)["data"]
    decompressed, names, normalization_features, original_shape = helper.decompress(
        model_path,
        input_path=compressed_output_path,
        input_path_deltas=os.path.join(
            output_path, "compressed_output", "compressed_deltas.npz.gz"
        ),
        input_batch_index=os.path.join(
            output_path,
            "compressed_output",
            "compressed_batch_index_metadata.npz.gz",
        ),
        config=config,
        output_path=output_path,
    )
    # if verbose:
    #    print(f"Model used: {model_name}")

    # if hasattr(config, "convert_to_blocks") and config.convert_to_blocks:
    #     print(
    #         "Converting Blocked Data into Standard Format. Old Shape - ",
    #         decompressed.shape,
    #         "Target Shape - ",
    #         data_before.shape,
    #     )
    #     if config.model_type == "dense":
    #         decompressed = decompressed.reshape(
    #             data_before.shape[0], data_before.shape[1], data_before.shape[2]
    #         )
    #     else:
    #         decompressed = decompressed.reshape(
    #             data_before.shape[0], 1, data_before.shape[1], data_before.shape[2]
    #         )

    if config.apply_normalization:
        print("Un-normalizing...")
        normalization_features = np.load(
            os.path.join(
                "/".join(output_path.split("/")[:-1]),
                "normalization_features.npy",
            ),
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

    end = time.time()
    print("Decompression took:", f"{(end - start) / 60:.3} minutes")

    return decompressed, names, original_shape

    # if config.extra_compression:
    #     if verbose:
    #         print("Extra compression selected")
    #         print(
    #             f"Saving decompressed file to {os.path.join(output_path, 'decompressed_output', 'decompressed.npz')}"
    #         )
    #     np.savez_compressed(
    #         os.path.join(output_path, "decompressed_output", "decompressed.npz"),
    #         data=decompressed,
    #         names=names,
    #     )
    # else:
    #     np.savez(
    #         os.path.join(output_path, "decompressed_output", "decompressed.npz"),
    #         data=decompressed,
    #         names=names,
    #     )
