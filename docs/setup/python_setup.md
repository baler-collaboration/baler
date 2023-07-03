## Python Setup
A setup for windows is available [here](documentation/setup/python_setup_windows.md)

For some Linux users, disable the KDE keyring
```console
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

1. Install [poetry](https://python-poetry.org/) for managing the python environment. 

2. Clone this repository and move into the project
```console
git clone https://github.com/baler-collaboration/baler.git && cd baler
```

3. Install dependencies using Poetry
```console
poetry install
```

4. Try the installation with `poetry run python -m baler`, this should give the following output:

```console
usage: baler.py [-h] --mode MODE --project WORKSPACE PROJECT [--verbose]
baler.py: error: the following arguments are required: --mode, --project
```

## Working Example with Python

Here we provide some instructions for our working examples.

### Computational Fluid Dynamics Example

#### Training ####
To train the autoencoder to compress your data, you run the following command. The config file `./workspaces/CFD_workspace/CFD_project_v1/config/CFD_project_v1_config.py`. This details the path of the data, the number of epochs, and all the other training parameters.
```console
poetry run python baler --project CFD_workspace CFD_project_animation --mode train
```

#### Compressing ####
To use the derived model for compression, you can now choose ``--mode compress``, which can be run as

```console
poetry run python -m baler --project CFD_workspace CFD_project_animation --mode compress
```
This will output a compressed file called "compressed.pickle", and this is the latent space representation of the input dataset. It will also output cleandata_pre_comp.pickle which is just the exact data being compressed.

#### Decompressing ####
To decompress the compressed file, we choose --mode decompress and run:
```console
poetry run python -m baler --project CFD_workspace CFD_project_animation --mode decompress
```

#### Plotting ####
To plot the difference of your variables before compression and after decompression, we can use the following command to generate a .pdf document under ``./workspaces/firstWorkspace/firstProject/output/plotting/comparison.pdf``

```console
poetry run python -m baler --project CFD_workspace CFD_project_animation --mode plot
```

### High Energy Physics Example ###
To run our High Energy Physics using CMS data (DOI:10.7483/OPENDATA.CMS.KL8H.HFVH) follow the above instructions but replace `--project CFD_workspace CFD_project_animation` with `--project CMS_workspace CMS_project_v1`

## New project ##
To create the folder structure and a skeleton config for your own project:
```console
poetry run python -m baler --project firstWorkspace firstProject --mode newProject
```