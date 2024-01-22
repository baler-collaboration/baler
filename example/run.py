import sys

sys.path.append("../src/")

# import baler
import baler_compressor.config as config_module
import baler_compressor.trainer as trainer_module
import baler_compressor.compressor as compressor_module
import baler_compressor.decompressor as decompressor_module
import baler_compressor.helper as baler_helper

sys.path.append("../../baler-models/")
import dense_demo as dense_demo_module

# import helper for plotting
import helper

# import others
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm


def define_config():
    # Initialize config
    config = config_module.Config

    # Define config
    config.input_path = "input/exafel_1.npz"
    config.output_path = "output/"

    config = config_module.Config
    config.compression_ratio = 1000
    config.epochs = 10
    config.early_stopping = False
    config.early_stopping_patience = 100
    config.min_delta = 0
    config.lr_scheduler = True
    config.lr_scheduler_patience = 50
    # config.model_name = "dense_demo"
    config.model = dense_demo_module.dense_demo
    config.model_type = "dense"
    config.custom_norm = True
    config.l1 = True
    config.reg_param = 0.001
    config.RHO = 0.05
    config.lr = 0.001
    config.batch_size = 75
    config.test_size = 0.2
    config.data_dimension = 2
    config.apply_normalization = False
    config.deterministic_algorithm = False
    config.compress_to_latent_space = False
    config.convert_to_blocks = [1, 150, 150]
    config.verbose = False

    # FPGA stuff
    config.number_of_columns = 22500  # FIXME: this is doesn't need to be hardcoded
    config.latent_space_size = 225  # FIXME: this is doesn't need to be hardcoded
    config.default_reuse_factor = 1
    config.default_precision = "ap_fixed<16,8>"
    config.Strategy = "latency"
    config.Part = "xcvu9p-flga2104-2L-e"
    config.ClockPeriod = 5
    config.IOType = "io_parallel"
    config.InputShape = (1, 16)
    config.ProjectName = "tiny_test_model"
    config.OutputDir = "workspaces/FPGA_compression_workspace/first_FPGA_Compression_project/output/hls4ml"
    config.InputData = None
    config.OutputPredictions = None
    config.csim = False
    config.synth = True
    config.cosim = False
    config.export = False

    return config


def train(config):
    # Run training
    model, normalization_features, loss_data = trainer_module.run(
        config.input_path, config
    )
    torch.save(model.state_dict(), config.output_path + "compressed_output/model.pt")
    np.save(
        config.output_path + "compressed_output/normalization_features.npy",
        normalization_features,
    )
    helper.loss_plot(loss_data[0], config.output_path, config)


def compress(config):
    # Run compression
    normalization_features = np.load(
        config.output_path + "compressed_output/normalization_features.npy"
    )
    compressed, names, original_shape = compressor_module.run(
        config.input_path,
        config.output_path + "compressed_output/model.pt",
        normalization_features,
        config,
    )

    # Save compressed file to disk
    np.savez_compressed(
        config.output_path + "compressed_output/compressed.npz",
        data=compressed,
        names=names,
        normalization_features=normalization_features,
        original_shape=original_shape,
    )
    # np.save(config.output_path + "loss_data.npy", loss_data)


def decompress(config):
    # Run decompression
    decompressed, names, original_shape = decompressor_module.run(
        config.output_path + "compressed_output/model.pt",
        config.output_path + "compressed_output/compressed.npz",
        config,
    )

    # Save decompressed file to disk
    np.savez(
        config.output_path + "decompressed_output/decompressed.npz",
        data=decompressed,
        names=names,
    )


def plot(config):
    data = np.load(config.input_path)["data"]
    data_decompressed = np.load(
        config.output_path + "decompressed_output/decompressed.npz"
    )["data"].reshape(data.shape[0], data.shape[1], data.shape[2])

    print("Making GIF: ")
    with imageio.get_writer(config.output_path + "/movie.gif", mode="I") as writer:
        for i in tqdm(range(0, 74)):
            figg = helper.plot2D(data[i], data_decompressed[i])
            image_name = config.output_path + f"plot_output/{i}.png"
            plt.savefig(image_name)
            image = imageio.imread(image_name)
            writer.append_data(image)


def FPGA(config):
    baler_helper.perform_hls4ml_conversion(config.output_path, config)


def main():
    config = define_config()

    if sys.argv[1] == "train":
        train(config)
    elif sys.argv[1] == "compress":
        compress(config)
    elif sys.argv[1] == "decompress":
        decompress(config)
    elif sys.argv[1] == "plot":
        plot(config)
    elif sys.argv[1] == "FPGA":
        FPGA(config)
    elif sys.argv[1] == "all":
        train(config)
        compress(config)
        decompress(config)
        plot(config)
        # FPGA(config)
    else:
        print("Unknown argument")


if __name__ == "__main__":
    main()
