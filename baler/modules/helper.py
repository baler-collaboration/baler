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

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
import math
from math import ceil
import gzip

from tqdm import tqdm

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from modules import training, plotting, data_processing, diagnostics


def get_arguments():
    """Determines the arguments one is able to apply in the command line when running Baler. Use `--help` to see what
    options are available.

    Returns: .py, string, folder: `.py` file containing the config options, string determining what mode to run,
    projects directory where outputs go.
    """
    parser = argparse.ArgumentParser(
        prog="baler",
        description=(
            "Baler is a machine learning based compression tool for big data.\n\n"
            "Baler has three running modes:\n\n"
            '\t1. Derivation: Using a configuration file and a "small" input dataset, Baler derives a '
            "machine learning model optimized to compress and decompress your data.\n\n"
            "\t2. Compression: Using a previously derived model and a large input dataset, Baler compresses "
            "your data and outputs a smaller compressed file.\n\n"
            "\t3. Decompression: Using a previously compressed file as input and a model, Baler decompresses "
            "your data into a larger file."
        ),
        epilog="Enjoy!",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="newProject, train, compress, decompress, plot, info",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        nargs=2,
        metavar=("WORKSPACE", "PROJECT"),
        help="Specifies workspace and project.\n"
        "e.g. --project CFD firstTry"
        ", specifies workspace 'CFD' and project 'firstTry'\n\n"
        "When combined with newProject mode:\n"
        "  1. If workspace and project exist, take no action.\n"
        "  2. If workspace exists but project does not, create project in workspace.\n"
        "  3. If workspace does not exist, create workspace directory and project.",
    )
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Verbose mode"
    )
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    workspace_name = args.project[0]
    project_name = args.project[1]
    config_path = (
        f"workspaces.{workspace_name}.{project_name}.config.{project_name}_config"
    )

    if args.mode == "newProject":
        config = None
    else:
        config = Config
        importlib.import_module(config_path).set_config(config)

    return (
        config,
        args.mode,
        workspace_name,
        project_name,
        args.verbose,
    )


def create_new_project(
    workspace_name: str,
    project_name: str,
    verbose: bool = False,
    base_path: str = "workspaces",
) -> None:
    """Creates a new project directory output subdirectories and config files within a workspace.

    Args:
        workspace_name (str): Creates a workspace (dir) for storing data and projects with this name.
        project_name (str): Creates a project (dir) for storing configs and outputs with this name.
        verbose (bool, optional): Whether to print out the progress. Defaults to False.
    """

    # Create full project path
    workspace_path = os.path.join(base_path, workspace_name)
    project_path = os.path.join(base_path, workspace_name, project_name)
    if os.path.exists(project_path):
        print(f"The workspace and project ({project_path}) already exists.")
        return
    os.makedirs(project_path)

    # Create required directories
    required_directories = [
        os.path.join(workspace_path, "data"),
        os.path.join(project_path, "config"),
        os.path.join(project_path, "output", "compressed_output"),
        os.path.join(project_path, "output", "decompressed_output"),
        os.path.join(project_path, "output", "plotting"),
        os.path.join(project_path, "output", "training"),
    ]

    if verbose:
        print(f"Creating project {project_name} in workspace {workspace_name}...")
    for directory in required_directories:
        if verbose:
            print(f"Creating directory {directory}...")
        os.makedirs(directory, exist_ok=True)

    # Populate default config
    with open(
        os.path.join(project_path, "config", f"{project_name}_config.py"), "w"
    ) as f:
        f.write(create_default_config(workspace_name, project_name))


@dataclass
class Config:
    """Defines a configuration dataclass"""

    input_path: str
    compression_ratio: float
    epochs: int
    early_stopping: bool
    early_stoppin_patience: int
    lr_scheduler: bool
    lr_scheduler_patience: int
    min_delta: int
    model_name: str
    model_type = str
    custom_norm: bool
    l1: bool
    reg_param: float
    RHO: float
    lr: float
    batch_size: int
    test_size: float
    data_dimension: int
    intermittent_model_saving: bool
    separate_model_saving: bool
    intermittent_saving_patience: int
    mse_avg: bool
    mse_sum: bool
    emd: bool
    l1: bool
    deterministic_algorithm: bool


def create_default_config(workspace_name: str, project_name: str) -> str:
    """Creates a default config file for a project.
    Args:
        workspace_name (str): Name of the workspace.
        project_name (str): Name of the project.
    Returns:
        str: Default config file.
    """

    return f"""
# === Configuration options ===

def set_config(c):
    c.input_path                   = "workspaces/{workspace_name}/data/{project_name}_data.npz"
    c.data_dimension               = 1
    c.compression_ratio            = 2.0
    c.apply_normalization          = True
    c.model_name                   = "AE"
    c.model_type                    = "dense"
    c.epochs                       = 5
    c.lr                           = 0.001
    c.batch_size                   = 512
    c.early_stopping               = True
    c.lr_scheduler                 = True




# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler_patience        = 50
    c.custom_norm                  = False
    c.reg_param                    = 0.001
    c.RHO                          = 0.05
    c.test_size                    = 0
    # c.number_of_columns            = 24
    # c.latent_space_size            = 12
    c.extra_compression            = False
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 100
    c.mse_avg                      = False
    c.mse_sum                      = True
    c.emd                          = False
    c.l1                           = True
    c.activation_extraction        = False
    c.deterministic_algorithm      = True
    c.separate_model_saving        = False

"""


def model_init(model_name: str):
    """Calls `data_processing.initialise_model`.

    Args:
        model_name (string): The name of the model you wish to initialize

    Returns:
       nn.Module : The initialized model
    """
    # This is used when we don't have saved model parameters.
    model_object = data_processing.initialise_model(model_name)
    return model_object


def numpy_to_tensor(data):
    """Converts ndarray to torch.Tensors.

    Args:
        data (ndarray): The data you wish to convert from ndarray to torch.Tensor.

    Returns:
        torch.Tensor: Your data as a tensor
    """
    return torch.from_numpy(data)


def normalize(data, custom_norm):
    """Applies `data_processing.normalize()` along every axis of given data

    Args:
        data (ndarray): Data you wish to normalize
        custom_norm (boolean): Wether or not you wish to use MinMax normalization

    Returns:
        ndarray: Normalized data
    """
    data = np.apply_along_axis(
        data_processing.normalize, axis=0, arr=data, custom_norm=custom_norm
    )
    return data


def process(
    input_path,
    custom_norm,
    test_size,
    apply_normalization,
    convert_to_blocks,
    verbose,
):
    """Loads the input data into a ndarray, splits it into train/test splits and normalizes if chosen.

    Args:
        input_path (string): Path to input data
        custom_norm (boolean): `False` if you want to use MinMax normalization. `True` otherwise.
        test_size (float): How much of your data to use as validation.
        apply_normalization (boolean): `True` if you want to normalize. `False` if you don't want to normalize.

    Returns: ndarray, ndarray, ndarray: Array with the train set, array with the test set and array with the
    normalization features.
    """
    loaded = np.load(input_path)
    data = loaded["data"]

    if verbose:
        print("Original Dataset Shape - ", data.shape)

    original_shape = data.shape

    if convert_to_blocks:
        data = data_processing.convert_to_blocks_util(convert_to_blocks, data)

    normalization_features = data_processing.find_minmax(data)
    if apply_normalization:
        print("Normalizing the data...")
        data = normalize(data, custom_norm)
    if not test_size:
        train_set = data
        test_set = train_set
    else:
        train_set, test_set = train_test_split(
            data, test_size=test_size, random_state=1
        )

    return (train_set, test_set, normalization_features, original_shape)


def renormalize(data, true_min_list, feature_range_list):
    """Calls `data_processing.renormalize_func()`.

    Args:
        data (ndarray): Data you wish to un-normalize
        true_min_list (ndarray): List of column minimums
        feature_range_list (ndarray): List of column feature ranges

    Returns:
        ndarray: Un-normalized array
    """
    return data_processing.renormalize_func(data, true_min_list, feature_range_list)


def train(model, number_of_columns, train_set, test_set, project_path, config):
    """Calls `training.train()`

    Args:
        model (modelObject): The model you wish to train
        number_of_columns (int): Amount of columns in the initial dataset
        train_set (ndarray): Array consisting of the train set
        test_set (ndarray): Array consisting of the test set
        project_path (string): Path to the project directory
        config (dataClass): Base class selecting user inputs

    Returns:
        _type_: _description_
    """
    return training.train(
        model, number_of_columns, train_set, test_set, project_path, config
    )


def plotter(output_path, config):
    """Calls `plotting.plot()`

    Args:
        output_path (string): Path to the output directory
        config (dataClass): Base class selecting user inputs

    """

    plotting.plot(output_path, config)
    print("=== Done ===")
    print("Your plots are available in:", os.path.join(output_path, "plotting"))


def loss_plotter(path_to_loss_data, output_path, config):
    """Calls `plotting.loss_plot()`

    Args:
        path_to_loss_data (string): Path to the values for the loss plot
        output_path (string): Path to output the data
        config (dataClass): Base class selecting user inputs

    Returns:
        .pdf file: Plot containing the loss curves
    """
    return plotting.loss_plot(path_to_loss_data, output_path, config)


def model_saver(model, model_path):
    """Calls `data_processing.save_model()`

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.

    Returns:
        .pt file: `.pt` File containing the model state dictionary
    """
    return data_processing.save_model(model, model_path)


def encoder_decoder_saver(model, encoder_path, decoder_path):
    """Calls `data_processing.encoder_saver` and `data_processing.decoder_saver`

    Args:
        model (nn.Module): The PyTorch model to save.
        model_path (str): String defining the models save path.

    Returns:
        .pt file: `.pt` File containing the encoder state dictionary
        .pt file: `.pt` File containing the decoder state dictionary

    """
    return data_processing.encoder_saver(
        model, encoder_path
    ), data_processing.decoder_saver(model, decoder_path)


def detacher(tensor):
    """Detaches a given tensor to ndarray

    Args:
        tensor (torch.Tensor): The PyTorch tensor one wants to convert to a ndarray

    Returns:
        ndarray: Converted torch.Tensor to ndarray
    """
    return tensor.cpu().detach().numpy()


def get_device():
    """Returns the appropriate processing device. IF cuda is available it returns "cuda:0"
        Otherwise it returns "cpu"

    Returns:
        _type_: Device string, either "cpu" or "cuda:0"
    """
    device = None
    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu"
        device = torch.device(dev)
    return device


def save_error_bounded_requirement(config, decoded_output, data_batch):
    rms_pred_error = (
        np.divide(
            np.subtract(decoded_output, data_batch),
            data_batch,
        )
        * 100
    )

    # Ignoring RMS Undefind Values because Ground Truth is Zero
    rms_pred_error[
        (rms_pred_error == np.inf)
        | (rms_pred_error == -np.inf)
        | (rms_pred_error == np.nan)
    ] = 0.0
    rms_pred_error_index = np.where(
        abs(rms_pred_error) > config.error_bounded_requirement
    )
    rows_idx, col_idx = rms_pred_error_index
    if len(rows_idx) > 0 and len(col_idx) > 0:
        rms_pred_error_exceeding_error_bound = np.subtract(
            decoded_output,
            data_batch,
            dtype=np.float16,
        )
        deltas = []
        for i in range(len(rows_idx)):
            deltas.append(rms_pred_error_exceeding_error_bound[rows_idx[i]][col_idx[i]])
    return deltas, rms_pred_error_index


def compress(model_path, config):
    """Function which performs the compression of the input file. In order to compress, you must have a dataset whose
    path is determined by `input_path` in the `config`. You also need a trained model from path `model_path`. The
    model path is then used to initialize the model used for compression. The data is then converted into a
    `torch.tensor` and then passed through `model.encode`.

    Args:
        model_path (string): Path to where the model is located
        config (dataClass): Base class selecting user inputs

    Raises:
        NameError: Baler currently only supports 1D (e.g. HEP) or 2D (e.g. CFD) data as inputs.

    Returns:
        torch.Tensor: Compressed data as PyTorch tensor
    """

    # Loads the data and applies normalization if config.apply_normalization = True
    loaded = np.load(config.input_path)
    data_before = loaded["data"]
    original_shape = data_before.shape

    if hasattr(config, "convert_to_blocks") and config.convert_to_blocks:
        data_before = data_processing.convert_to_blocks_util(
            config.convert_to_blocks, data_before
        )

    if config.apply_normalization:
        print("Normalizing...")
        data = normalize(data_before, config.custom_norm)
    else:
        data = data_before
    number_of_columns = 0
    try:
        n_features = 0
        if config.data_dimension == 1:
            column_names = np.load(config.input_path)["names"]
            number_of_columns = len(column_names)
            config.latent_space_size = ceil(
                number_of_columns / config.compression_ratio
            )
            config.number_of_columns = number_of_columns
            n_features = number_of_columns
        elif config.data_dimension == 2:
            if config.model_type == "dense":
                number_of_rows = data.shape[1]
                config.number_of_columns = data.shape[2]
                n_features = number_of_rows * config.number_of_columns
            else:
                number_of_rows = original_shape[1]
                config.number_of_columns = original_shape[2]
                n_features = config.number_of_columns
            config.latent_space_size = ceil(
                (number_of_rows * config.number_of_columns) / config.compression_ratio
            )
        else:
            raise NameError(
                "Data dimension can only be 1 or 2. Got config.data_dimension = "
                + str(config.data_dimension)
            )
    except AttributeError:
        number_of_columns = config.number_of_columns
        latent_space_size = config.latent_space_size
        print(f"{number_of_columns} -> {latent_space_size} dimensions")

    # Initialise and load the model correctly.
    latent_space_size = config.latent_space_size
    bs = config.batch_size
    device = get_device()
    model_object = data_processing.initialise_model(config.model_name)
    model = data_processing.load_model(
        model_object,
        model_path=model_path,
        n_features=n_features,
        z_dim=config.latent_space_size,
    )
    model.eval()

    if config.data_dimension == 2:
        if config.model_type == "convolutional" and config.model_name == "Conv_AE_3D":
            data_tensor = torch.tensor(data, dtype=torch.float32).view(
                data.shape[0] // bs, 1, bs, data.shape[1], data.shape[2]
            )
        elif config.model_type == "convolutional":
            data_tensor = torch.tensor(data, dtype=torch.float32).view(
                data.shape[0], 1, data.shape[1], data.shape[2]
            )
        elif config.model_type == "dense":
            data_tensor = torch.tensor(data, dtype=torch.float32).view(
                data.shape[0], data.shape[1] * data.shape[2]
            )
    elif config.data_dimension == 1:
        data_tensor = torch.tensor(data, dtype=torch.float64)

    # Batching data to avoid memory leaks
    data_dl = DataLoader(
        data_tensor,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )

    # Perform compression
    error_bound_batch, error_bound_deltas, error_bound_index, compressed = (
        [],
        [],
        [],
        [],
    )

    with torch.no_grad():
        for idx, data_batch in enumerate(tqdm(data_dl)):
            data_batch = data_batch.to(device)

            compressed_output = model.encode(data_batch)

            if config.save_error_bounded_deltas:
                decoded_output = model.decode(compressed_output)
                decoded_output = detacher(decoded_output)
                deltas_compressed = 0
            # Converting back to numpyarray
            compressed_output = detacher(compressed_output)
            data_batch = detacher(data_batch)

            if config.save_error_bounded_deltas:
                (
                    deltas,
                    rms_pred_error_index,
                ) = save_error_bounded_requirement(config, decoded_output, data_batch)
                if len(rms_pred_error_index) > 0:
                    error_bound_batch.append(idx)
                    error_bound_deltas.append(deltas)
                    error_bound_index.append(rms_pred_error_index)
                    deltas_compressed += len(rms_pred_error_index[0])

            if idx == 0:
                compressed = compressed_output
            else:
                compressed = np.concatenate((compressed, compressed_output))

    if config.save_error_bounded_deltas:
        print("Total Deltas Found - ", deltas_compressed)

    return (compressed, error_bound_batch, error_bound_deltas, error_bound_index)


def decompress(
    model_path,
    input_path,
    input_path_deltas,
    input_batch_index,
    model_name,
    config,
    output_path,
    original_shape,
):
    """Function which performs the decompression of the compressed file. In order to decompress, you must have a
    compressed file, whose path is determined by `input_path`, a model from path `model_path` and a model_name. The
    model path and model names are used to initialize the model used for decompression. The data is then converted
    into a `torch.tensor` and then passed through `model.decode`.

    Args:
        model_path (string): Path to where the model is located
        input_path (string): Path to the data you want to decompress
        model_name (string): Name of trained model from which you want to use decode
        config (dataClass): Base class selecting user inputs

    Returns: torch.tensor, ndarray, ndarray: decompressed data as tensor, ndarray of column names, ndarray of
    normalization features
    """

    # Load the data & define necessary variables
    loaded = np.load(input_path)
    data = loaded["data"]
    names = loaded["names"]
    normalization_features = loaded["normalization_features"]

    if config.model_type == "convolutional":
        final_layer_details = np.load(
            os.path.join(output_path, "training", "final_layer.npy"), allow_pickle=True
        )

    if config.save_error_bounded_deltas:
        loaded_deltas = np.load(
            gzip.GzipFile(input_path_deltas, "r"), allow_pickle=True
        )
        loaded_batch_indexes = np.load(
            gzip.GzipFile(input_batch_index, "r"), allow_pickle=True
        )
        error_bound_batch = loaded_batch_indexes[0]
        error_bound_deltas = loaded_deltas
        error_bound_index = loaded_batch_indexes[1]
        deltas_added = 0

    model_name = config.model_name
    latent_space_size = len(data[0])
    bs = config.batch_size
    model_dict = torch.load(str(model_path), map_location=get_device())
    if config.data_dimension == 2 and config.model_type == "dense":
        number_of_columns = int((len(model_dict[list(model_dict.keys())[-1]])))
    else:
        number_of_columns = len(model_dict[list(model_dict.keys())[-1]])

    # Initialise and load the model correctly.
    device = get_device()
    model_object = data_processing.initialise_model(model_name)
    model = data_processing.load_model(
        model_object,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=latent_space_size,
    )
    model.eval()

    if config.model_type == "convolutional":
        model.set_final_layer_dims(final_layer_details)

    # Load the data, convert to tensor and batch it to avoid memory leaks
    data_tensor = torch.from_numpy(data).to(device)
    data_dl = DataLoader(
        data_tensor,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )

    # Perform Decompression
    decompressed = []
    with torch.no_grad():
        for idx, data_batch in enumerate(tqdm(data_dl)):
            data_batch = data_batch.to(device)

            out = model.decode(data_batch).to(device)
            # Converting back to numpyarray
            out = detacher(out)
            if config.save_error_bounded_deltas:
                if idx in error_bound_batch:
                    # Error Bounded Deltas added to Decompressed output
                    delta_idx = np.where(error_bound_batch == idx)
                    deltas = error_bound_deltas[delta_idx][0]
                    delta_index = error_bound_index[delta_idx][0]
                    row_idx, col_idx = delta_index

                    for i in range(len(row_idx)):
                        out[row_idx[i]][col_idx[i]] -= deltas[i]
                        deltas_added += 1

            if idx == 0:
                decompressed = out
            else:
                decompressed = np.concatenate((decompressed, out))

    if config.save_error_bounded_deltas:
        print("Total Deltas Added - ", deltas_added)

    if config.data_dimension == 2 and config.model_type == "dense":
        decompressed = decompressed.reshape(
            (len(decompressed), original_shape[1], original_shape[2])
        )

    return decompressed, names, normalization_features


def diagnose(input_path: str, output_path: str) -> None:
    """Calls diagnostics.diagnose()

    Args:
        input_path (str): path to the np.array contataining the activations values
        output_path (str): path to store the diagnostics pdf
    """
    diagnostics.diagnose(input_path, output_path)


def perform_hls4ml_conversion(output_path, config):
    """Function which performs the conversion of the model to FPGA architecture using hls4ml. In order to convert, you must have a trained model.

    The output hls4ml project will be located in OutputDir defined in the config file.
    The model path is determined by the `output_path`. The 'output_path' should point to the output directory inside the project directory.

    Before running this function:
    1. Install Vivado 2020.1 and add in to path.
    2. Install hls4ml and tensorflow to the projects virtual environment.
    3. Add the following configuration options to the projects config:

        c.default_reuse_factor
        c.default_precision
        c.Strategy
        c.Part
        c.ClockPeriod
        c.IOType
        c.InputShape
        c.ProjectName
        c.OutputDir
        c.InputData
        c.OutputPredictions
        c.csim
        c.synth
        c.cosim
        c.export

        This function was tested with the following configuration values:

        c.default_reuse_factor         = 1
        c.default_precision            = "ap_fixed<16,8>"
        c.Strategy                     = "latency"
        c.Part                         = "xcvu9p-flga2104-2L-e"
        c.ClockPeriod                  = 5
        c.IOType                       = "io_parallel"
        c.InputShape                   = (1,16)
        c.ProjectName                  = "tiny_test_model"
        c.OutputDir                    = "workspaces/FPGA_compression_workspace/first_FPGA_Compression_project/output/hls4ml"
        c.InputData                    = None
        c.OutputPredictions            = None
        c.csim                         = False
        c.synth                        = True
        c.cosim                        = False
        c.export                       = False

        Naming is based on the names used by hls4ml for easier reference. For more details please see hls4ml documentation.

    4. The model to transform should be defined similarly to FPGA_prototype_model. Meaning:
        a. Activations should be defined as layers and not functions.
        b. If the model is defined as a class, each layer should be defined with an attribute. An attribute for the whole model instead of for each layer separately might cause errors in hls4ml compile function.
        c. Not all types of layers are supported in hls4ml, check hls4ml for the supported layers.



    Args:
    output_path (string): Path to the output directory inside the project directory.
    config (dataClass): Base class selecting user inputs

    Returns: None


    """

    import hls4ml

    model_path = os.path.join(output_path, "compressed_output", "model.pt")

    model_object = data_processing.initialise_model(config.model_name)
    model = data_processing.load_model(
        model_object,
        model_path=model_path,
        n_features=config.number_of_columns,
        z_dim=config.latent_space_size,
    )
    model.to("cpu")

    hls_config = hls4ml.utils.config_from_pytorch_model(
        model,
        granularity="name",
        default_reuse_factor=config.default_reuse_factor,
        default_precision=config.default_precision,
    )

    hls_config["Model"]["Strategy"] = config.Strategy

    cfg = hls4ml.converters.create_config(backend="Vivado")
    cfg["Part"] = config.Part
    cfg["ClockPeriod"] = config.ClockPeriod
    cfg["IOType"] = config.IOType
    cfg["HLSConfig"] = hls_config
    cfg["PytorchModel"] = model
    cfg["InputShape"] = config.InputShape
    cfg["OutputDir"] = config.OutputDir

    if config.InputData and config.OutputPredictions:
        cfg["InputData"] = config.InputData
        cfg["OutputPredictions"] = config.OutputPredictions

    hls_model = hls4ml.converters.pytorch_to_hls(cfg)
    hls_model.config.config["ProjectName"] = config.ProjectName

    hls_model.compile()

    hls_model.build(
        csim=config.csim, synth=config.synth, cosim=config.cosim, export=config.export
    )

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features).double())
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order).double()
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
