import argparse
import os
import pickle

import numpy
import pandas
import torch

from modules import training, plotting, data_processing


def get_arguments():
    parser = argparse.ArgumentParser(
        prog="baler.py",
        description="""Baler is a machine learning based compression tool for big data.\n
Baler has three running modes:\n
\t1. Derivation: Using a configuration file and a "small" input dataset, Baler derives a machine learning model optimized to compress and decompress your data.\n
\t2. Compression: Using a previously derived model and a large input dataset, Baler compresses your data and outputs a smaller compressed file.\n
\t3. Decompression: Using a previously compressed file as input and a model, Baler decompresses your data into a larger file.""",
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
        args.config = ""
    else:
        config_path = f"projects/{args.project}/config.json"
        args.config = data_processing.import_config(config_path)

    return args.config, args.mode, args.project


def create_new_project(project_name: str, base_path: str = "projects") -> None:
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
    with open(os.path.join(project_path, "config.json"), "w") as f:
        f.write(create_default_config())
    for directory in required_directories:
        os.makedirs(os.path.join(project_path, directory))


def create_default_config() -> str:
    return """
{
    "epochs" : 5,
    "early_stopping": true,
    "lr_scheduler" : false,
    "patience" : 100,
    "min_delta" : 0,
    "model_name" : "george_SAE",
    "custom_norm" : false,
    "l1" : true,
    "reg_param" : 0.001,
    "RHO" : 0.05,
    "lr" : 0.001,
    "batch_size" : 512,
    "save_as_root" : true,
    "cleared_col_names":["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"],
    "test_size" : 0.15,
    "Branch" : "Events",
    "Collection": "recoGenJets_slimmedGenJets__PAT.",
    "Objects": "recoGenJets_slimmedGenJets__PAT.obj",
    "number_of_columns" : 8,
    "latent_space_size" : 4,
    "dropped_variables":    [
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fX",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fY",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fZ",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.qx3_",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.pdgId_",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.status_",
        "recoGenJets_slimmedGenJets__PAT.obj.mJetArea",
        "recoGenJets_slimmedGenJets__PAT.obj.mPileupEnergy",
        "recoGenJets_slimmedGenJets__PAT.obj.mPassNumber"
    ],
    "input_path":"data/firstProject/cms_data.root"
}"""


def to_pickle(data, path):
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def model_init(config):
    # This is used when we don't have saved model parameters.
    ModelObject = data_processing.initialise_model(config=config)
    return ModelObject


def data_loader(data_path, config):
    return data_processing.load_data(data_path, config)


def numpy_to_tensor(data):
    if isinstance(data, pandas.DataFrame):
        data = data.to_numpy()

    return torch.from_numpy(data)


def normalize(data, config):
    data = numpy.apply_along_axis(
        data_processing.normalize, axis=0, arr=data, config=config
    )
    df = data_processing.numpy_to_df(data, config)
    return df


def process(data_path, config):
    df = data_processing.load_data(data_path, config)
    df = data_processing.clean_data(df, config)
    if config["energy"] == True:
        #    df = Concat_energy(df)
        df = convert_mass_to_energy(df)
    full_pre_norm = df
    normalization_features = data_processing.find_minmax(df)
    df = normalize(df, config)
    full_norm = df
    train_set, test_set = data_processing.split(
        df, test_size=config["test_size"], random_state=1
    )
    number_of_columns = len(data_processing.get_columns(df))
    assert (
        number_of_columns == config["number_of_columns"]
    ), f"The number of columns of dataframe is {number_of_columns}, config states {config['number_of_columns']}."
    return (
        train_set,
        test_set,
        number_of_columns,
        normalization_features,
        full_norm,
        full_pre_norm,
    )


def renormalize(data, true_min_list, feature_range_list):
    return data_processing.renormalize_func(data, true_min_list, feature_range_list)


def train(model, number_of_columns, train_set, test_set, project_path, config):
    return training.train(
        model, number_of_columns, train_set, test_set, project_path, config
    )


def plot(output_path, before, after):
    plotting.plot(output_path, before, after)


def loss_plotter(path_to_loss_data, output_path, config):
    return plotting.loss_plot(path_to_loss_data, output_path, config)


def model_saver(model, model_path):
    return data_processing.save_model(model, model_path)


def detach(tensor):
    return tensor.cpu().detach().numpy()


def compress(number_of_columns, model_path, input_path, config):
    # Initialise and load the model correctly.
    ModelObject = data_processing.initialise_model(config=config)
    model = data_processing.load_model(
        ModelObject,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=config["latent_space_size"],
    )

    # Give the encoding function the correct input as tensor
    data = data_loader(input_path, config)

    ## CHANGE BACK

    # data = data_processing.clean_data(data, config)
    data_before = numpy.array(data)
    data = normalize(data, config)
    data_tensor = numpy_to_tensor(data).to(model.device)

    compressed = model.encode(data_tensor)
    return compressed, data_before


def decompress(number_of_columns, model_path, input_path, config):
    # Initialise and load the model correctly.
    ModelObject = data_processing.initialise_model(config=config)
    model = data_processing.load_model(
        ModelObject,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=config["latent_space_size"],
    )

    # Load the data & convert to tensor
    data = data_loader(input_path, config)
    data_tensor = numpy_to_tensor(data).to(model.device)

    decompressed = model.decode(data_tensor)
    return decompressed


def to_root(data_path, config, save_path):
    # if '.pickle' in data_path[-8:]:
    if isinstance(data_path, pickle.Pickler):
        df, Names = data_processing.pickle_to_df(file_path=data_path, config=config)
        return data_processing.df_to_root(df, Names, save_path)
    elif isinstance(data_path, pandas.DataFrame):
        return data_processing.df_to_root(
            data_path, col_names=data_path.columns(), save_path=save_path
        )
    elif isinstance(data_path, numpy.ndarray):
        df = data_processing.numpy_to_df(data_path, config)
        df_names = df.columns
        return data_processing.df_to_root(df, col_names=df_names, save_path=save_path)


def get_device():
    device = None
    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu"
        device = torch.device(dev)
    return device


def Concat_energy(df):
    Energy_df = data_processing.compute_E(
        mass=df["recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fM"],
        eta=df[
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fEta"
        ],
        pt=df["recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fPt"],
    )
    concat_df = pandas.concat([df, Energy_df], axis=1)
    return concat_df


def convert_mass_to_energy(df):
    # Takes df with mass
    # mass_col_name = [col for col in df.columns if ".fM" in col]
    # pt_col_name = [col for col in df.columns if ".fPt" in col]
    # eta_col_name = [col for col in df.columns if ".fEta" in col]
    print(df.columns)
    mass = df["recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fM"]
    eta = df["recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fEta"]
    pt = df["recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fPt"]
    energy = data_processing.compute_E(mass=mass, eta=eta, pt=pt)
    df["recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fM"] = energy

    """
    def energy(mass):
        masspt = (
            df["recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fPt"]
            ** 2
            + mass**2
        )
        cosh = (
            numpy.cosh(
                df[
                    "recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fEta"
                ]
            )
        ) ** 2
        total = numpy.sqrt(masspt * cosh)
        return total

    df["recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fM"] = df[
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fM"
    ].apply(energy)
    
    """

    return df
