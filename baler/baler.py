import os
import time

import pandas as pd

import modules.helper as helper
import importlib


def main():
    config, mode, project_name = helper.get_arguments()
    project_path = f"projects/{project_name}/"
    if mode == "newProject":
        helper.create_new_project(project_name)
    elif mode == "pp":
        pre_processing(config.input_path, project_name)
    elif mode == "train":
        perform_training(config, project_path)
    elif mode == "plot":
        perform_plotting(project_path, config)
    elif mode == "compress":
        perform_compression(config, project_path)
    elif mode == "decompress":
        perform_decompression(config.save_as_root, config.model_name, project_path)
    elif mode == "info":
        print_info(project_path)


def pre_processing(input_path, project_name):
    importlib.import_module(
        f"projects.{project_name}.{project_name}_config"
    ).pre_processing(input_path)


def perform_training(config, project_path):
    (
        train_set,
        test_set,
        number_of_columns,
        normalization_features,
        cleared_col_names,
    ) = helper.process(config.input_path, config.test_size)
    train_set_norm = helper.normalize(train_set, config.custom_norm, cleared_col_names)
    test_set_norm = helper.normalize(test_set, config.custom_norm, cleared_col_names)
    try:
        config.latent_space_size = int(number_of_columns // config.compression_ratio)
        config.number_of_columns = number_of_columns
    except AttributeError:
        print(config.latent_space_size, config.number_of_columns)
        assert number_of_columns == config.number_of_columns

    device = helper.get_device()

    ModelObject = helper.model_init(config.model_name)
    model = ModelObject(
        device=device, n_features=number_of_columns, z_dim=config.latent_space_size
    )

    output_path = project_path + "training/"
    test_data_tensor, reconstructed_data_tensor = helper.train(
        model, number_of_columns, train_set_norm, test_set_norm, output_path, config
    )
    test_data = helper.detach(test_data_tensor)
    reconstructed_data = helper.detach(reconstructed_data_tensor)

    print("Un-normalzing...")
    start = time.time()
    test_data_renorm = helper.renormalize(
        test_data,
        normalization_features["True min"],
        normalization_features["Feature Range"],
    )
    reconstructed_data_renorm = helper.renormalize(
        reconstructed_data,
        normalization_features["True min"],
        normalization_features["Feature Range"],
    )
    end = time.time()
    print("Un-normalization took:", f"{(end - start) / 60:.3} minutes")

    helper.to_pickle(test_data_renorm, output_path + "before.pickle")
    helper.to_pickle(reconstructed_data_renorm, output_path + "after.pickle")
    # normalization_features.to_csv(project_path + "model/cms_normalization_features.csv")
    helper.model_saver(model, project_path + "model/model.pt")
    helper.to_pickle(
        cleared_col_names, project_path + "compressed_output/column_names.pickle"
    )


def perform_plotting(project_path, config):
    helper.plot(project_path)
    helper.loss_plotter(
        project_path + "training/loss_data.csv", project_path + "plotting/", config
    )


def perform_compression(config, project_path):
    print("Compressing...")
    start = time.time()
    compressed, data_before, cleared_col_names = helper.compress(
        model_path=project_path + "model/model.pt",
        config=config,
    )
    # Converting back to numpyarray
    compressed = helper.detach(compressed)
    end = time.time()

    print("Compression took:", f"{(end - start) / 60:.3} minutes")

    helper.to_pickle(compressed, project_path + "compressed_output/compressed.pickle")
    helper.to_pickle(
        data_before, project_path + "compressed_output/cleandata_pre_comp.pickle"
    )
    helper.to_pickle(
        cleared_col_names, project_path + "compressed_output/column_names.pickle"
    )


def perform_decompression(save_as_root, model_name, project_path):
    print("Decompressing...")
    cleared_col_names = helper.from_pickle(
        project_path + "compressed_output/column_names.pickle"
    )
    start = time.time()
    decompressed = helper.decompress(
        model_path=project_path + "model/model.pt",
        input_path=project_path + "compressed_output/compressed.pickle",
        model_name=model_name,
    )

    # Converting back to numpyarray
    decompressed = helper.detach(decompressed)
    normalization_features = pd.read_csv(
        project_path + "model/cms_normalization_features.csv"
    )

    decompressed = helper.renormalize(
        decompressed,
        normalization_features["True min"],
        normalization_features["Feature Range"],
    )
    end = time.time()
    print("Decompression took:", f"{(end - start) / 60:.3} minutes")

    # False by default
    if save_as_root:
        helper.to_root(
            decompressed,
            cleared_col_names,
            project_path + "decompressed_output/decompressed.root",
        )
        helper.to_pickle(
            decompressed, project_path + "decompressed_output/decompressed.pickle"
        )
    else:
        helper.to_pickle(
            decompressed, project_path + "decompressed_output/decompressed.pickle"
        )


def print_info(project_path):
    print(
        "================================== \n Information about your compression \n================================== "
    )

    pre_compression = project_path + "compressed_output/cleandata_pre_comp.pickle"
    compressed = project_path + "compressed_output/compressed.pickle"
    decompressed = project_path + "decompressed_output/decompressed.pickle"

    files = [pre_compression, compressed, decompressed]
    q = []
    for i in range(len(files)):
        q.append(os.stat(files[i]).st_size / (1024 * 1024))

    print(
        f"\nCompressed file is {round(q[1] / q[0], 2) * 100}% the size of the original\n"
    )
    print(f"File size before compression: {round(q[0], 2)} MB")
    print(f"Compressed file size: {round(q[1], 2)} MB")
    print(f"De-compressed file size: {round(q[2], 2),} MB")
    print(f"Compression ratio: {round(q[0] / q[1], 2)}")


if __name__ == "__main__":
    main()
