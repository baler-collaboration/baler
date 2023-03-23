import argparse
import importlib
import os
import sys
from dataclasses import dataclass

sys.path.append(os.getcwd())
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from modules import training, plotting, data_processing


def get_arguments():
    """Determines the arguments one is able to apply in the command line when running Baler. Use `--help` to see what options are avaliable.

    Returns:
        .py, string, folder: `.py` file containing the config options, string determining what mode to run, projects directory where outputs go.
    """
    parser = argparse.ArgumentParser(
        prog="baler.py",
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
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        help="train, compress, decompress, plot, info",
    )
    parser.add_argument(
        "--project", type=str, required=False, help="Name of new project"
    )

    args = parser.parse_args()
    if not args.mode or (args.mode != "newProject" and not args.project):
        parser.print_usage()
        exit(1)
    if args.mode == "newProject":
        config = None
    else:
        config = Config
        importlib.import_module(
            f"projects.{args.project}.{args.project}_config"
        ).set_config(config)
    return config, args.mode, args.project


def create_new_project(project_name: str, base_path: str = "projects") -> None:
    """Creates a new project directory with all necessary sub-directories and config files.

    Args:
        project_name (str): Determines what you want to call your new project as.
        base_path (str, optional): Defaults to "projects"
    """
    project_path = os.path.join(base_path, project_name)
    if os.path.exists(project_path):
        print(f"The project {project_path} already exists.")
        return

    required_directories = [
        "compressed_output",
        "decompressed_output",
        "plotting",
        "training",
        "model",
    ]
    os.makedirs(project_path)
    with open(os.path.join(project_path, f"{project_name}_config.py"), "w") as f:
        print(project_path)
        f.write(create_default_config(project_name))
    for directory in required_directories:
        os.makedirs(os.path.join(project_path, directory))


@dataclass
class Config:
    input_path: str
    compression_ratio: float
    epochs: int
    early_stopping: bool
    early_stoppin_patience: int
    lr_scheduler: bool
    lr_scheduler_patience: int
    min_delta: int
    model_name: str
    custom_norm: bool
    l1: bool
    reg_param: float
    RHO: float
    lr: float
    batch_size: int
    test_size: float
    data_dimension: int
    intermittent_model_saving: bool
    intermittent_saving_patience: int


def create_default_config(project_name: str) -> str:
    return f"""
def set_config(c):
    c.input_path                   = "data/{project_name}/{project_name}_data.npz"
    c.compression_ratio            = 2.0
    # c.number_of_columns            = 24
    # c.latent_space_size            = 12
    c.epochs                       = 5
    c.early_stopping               = True
    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler                 = False
    c.lr_scheduler_patience        = 100
    c.model_name                   = "AE"
    c.custom_norm                  = False
    c.l1                           = True
    c.reg_param                    = 0.001
    c.RHO                          = 0.05
    c.lr                           = 0.001
    c.batch_size                   = 512
    c.test_size                    = 0.15
    c.data_dimension               = 1
    c.apply_normalization          = True
    c.extra_compression            = False
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 100

"""


def model_init(model_name):
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
    """Converts ndarrays to torch.Tensors.

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


def process(input_path, custom_norm, test_size, apply_normalization):
    """Loads the input data into an ndarray, splits it into train/test splits and normalizes if chosen.

    Args:
        input_path (string): Path to input data
        custom_norm (boolean): `False` if you want to use MinMax normalization. `True` otherwise.
        test_size (float): How much of your data to use as validation.
        apply_normalization (boolean): `True` if you want to normalize. `False` if you don't want to normalize.

    Returns:
        ndarray, ndarray, ndarray: Array with the train set, array with the test set and array with the normalization features.
    """
    loaded = np.load(input_path)
    data = loaded["data"]
    names = loaded["names"]
    normalization_features = []

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

    return (
        train_set,
        test_set,
        normalization_features,
    )


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


def plotter(project_path, config):
    """Calls `plotting.plot()`

    Args:
        project_path (string): Path to the project directory
        config (dataClass): Base class selecting user inputs

    """

    plotting.plot(project_path, config)
    print("### Done ###")
    print("Your plots are available in:", project_path + "plotting/")


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


def detacher(tensor):
    """Detaches a given tensor to ndarray

    Args:
        tensor (torch.Tensor): The PyTorch tensor one wants to convert to a ndarray

    Returns:
        ndarray: Converted torch.Tensor to ndarray
    """
    return tensor.cpu().detach().numpy()


def get_device():
    device = None
    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu"
        device = torch.device(dev)
    return device


def compress(model_path, config):
    """Function which performs the compression of the input file. In order to compress, you must have a dataset whose path is
        determined by `input_path` in the `config`. You also need a trained model from path `model_path`. The model path is then used to initialize the model
        used for compression. The data is then converted into a `torch.tensor` and then passed through `model.encode`.

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
    if config.apply_normalization:
        print("Normalizing...")
        data = normalize(data_before, config.custom_norm)
    else:
        data = data_before
    number_of_columns = 0
    try:
        print("compression ratio:", config.compression_ratio)
        if config.data_dimension == 1:
            column_names = np.load(config.input_path)["names"]
            number_of_columns = len(column_names)
            config.latent_space_size = int(
                number_of_columns // config.compression_ratio
            )
            config.number_of_columns = number_of_columns
        elif config.data_dimension == 2:
            data = np.load(config.input_path)["data"]
            number_of_rows = data.shape[1]
            config.number_of_columns = data.shape[2]
            config.latent_space_size = int(
                (number_of_rows * number_of_columns) // config.compression_ratio
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
    device = get_device()
    model_object = data_processing.initialise_model(config.model_name)
    model = data_processing.load_model(
        model_object,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=latent_space_size,
    )

    # Give the encoding function the correct input as tensor
    if config.data_dimension == 2:
        data_tensor = (
            torch.from_numpy(data.astype("float32", casting="same_kind"))
            .to(device)
            .view(data.shape[0], 1, data.shape[1], data.shape[2])
        )
    elif config.data_dimension == 1:
        data_tensor = torch.from_numpy(data).to(device)

    compressed = model.encode(data_tensor)
    return compressed


def decompress(model_path, input_path, model_name):
    """Function which performs the decompression of the compressed file. In order to decompress, you must have a compressed file, whose path is
        determined by `input_path`, a model from path `model_path` and a model_name. The model path and model names are used to initialize the model
        used for decompression. The data is then converted into a `torch.tensor` and then passed through `model.decode`.

    Args:
        model_path (string): Path to where the model is located
        input_path (string): Path to the data you want to decompress
        model_name (string): Name of trained model from which you want to use decode

    Returns:
        torch.tensor, ndarray, ndarray: decompressed data as tensor, ndarray of column names, ndarray of normalization features
    """

    # Load the data & define necessary variables
    loaded = np.load(input_path)
    data = loaded["data"]
    names = loaded["names"]
    normalization_features = loaded["normalization_features"]
    latent_space_size = len(data[0])
    model_dict = torch.load(str(model_path))
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

    # Load the data & convert to tensor
    data_tensor = torch.from_numpy(data).to(device)

    # Decompress the data using the trained models decode function
    decompressed = model.decode(data_tensor)
    return decompressed, names, normalization_features
