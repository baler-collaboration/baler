# Introcution
Baler is a tool used to test the feasibility of compressing different types of scientific data using autoencoders.

The main object the user has produce before using baler is a configuration file. The configuration primarily contains:
* The path to the data which is going to be compressed
* The name of the autoencoder to be used
  - Either use a pre-made one or write your own, specific to your dataset, in `./modules/models.py`
* Pruning of the dataset, for example the dropping certain variables
* The number of epochs, batch size, learning rate etc.

When provided a configuration file, Baler has 4 main running modes:
* Train: Baler will train a machine learning model to compress and decompress your dataset, and then save the model, model weights, and normalization factors
* Plot: Baler will show the performance of the a trained network by plotting the difference between the original dataset and the compressed+decompressed dataset
* Compress: Baler will compress the dataset using the model derived in during the training mode
* Decompress: Baler will decompress the dataset using the model derived during the training mode

# Getting started #

## Windows
* First download and install the linux kernel update package for windows: https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package
* Install docker from docker.com
* Restart your computer and open docker Desktop
* Download this repository: https://github.com/baler-compressor/baler/archive/refs/heads/gpu_training.zip
* Open the downloaded file and extract it to your Desktop
* Press the windows key + r to open "run"
* Type in "cmd" in the window and hit enter to open the "terminal"
* Copy this command and paste it into the terminal, then hit enter
```console
cd Desktop\baler-main
```
* Paste this command into the terminal adn hit enter:
```console
docker run --mount type=bind,source=C:\Users\pekman\Desktop\baler-main\projects\,target=\projects ghcr.io/uomresearchit/baler:latest
```

## Poetry

Baler is currently packaged using [Poetry](https://python-poetry.org/ "Poetry"), a package manager for Python which helps with dependancy management. Installing and running using Poetry requires slight modifications to the usual Python commands, detailed [below](#installing-baler-dependancies-using-poetry).


Additionally, a Docker container is available which allows the use of Baler without worrying about dependencies or environment. Instructions for this usage are given [later in this README](#running-with-docker "Running with Docker").

## Example data ##

If you wish to use this README as a tutorial, you can use the following command to acquire example data, compatible with Baler and the configuration provided in this repository.

```console
mkdir data/firstProject; wget http://opendata.cern.ch/record/21856/files/assets/cms/mc/RunIIFall15MiniAODv2/ZprimeToTT_M-3000_W-30_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v1/10000/DAA238E5-29D6-E511-AE59-001E67DBE3EF.root -P ./data/firstProject/cms_data.root
```

When you have downloaded the example data, check the file is correct by computing the md5sum:

```console 
md5sum data/firstProject/cms_data.root 
> 28910642bf94e0fa9442bc804830f88b  data/firstProject/cms_data.root
```

## Running locally  ##

### Installing Baler dependencies using Poetry ###

First, if you do not already have Poetry installed on your system, acquire it in a manner appropriate for your system. For example, using pip:

```console
pip install poetry
```

Now, Poetry can be used to install the project dependencies. 

In the directory in which you have cloned this repository:

```console
poetry install
```

### Running Baler ###
Baler can be run locally using a virtual environment created by Poetry. This is achieved using the `poetry run` command.

#### Create New Project ####
Start by creating a new project directory. This will create the standardised directory structure needed, and create a blank config and output directories. `./projects/firstProject/config.json`.\
```console
poetry run python baler --mode=newProject --project=firstProject
```

#### Training ####
```console
poetry run python baler --project=firstProject --mode=train
```

This will most importantly, output: current_model.pt. This contains all necessary model parameters.

#### Compressing ####

To do something with this model, you can now choose --mode=compress, which can be ran as

```console
poetry run python baler --project=firstProject --mode=compress
```

This will output a compressed file called compressed.pickle, and this is the latent space representation of the input dataset. It will also output cleandata_pre_comp.pickle which is just the exact data being compressed.

#### Decompressing ####

To decompress the compressed file, we choose --mode=decompress and run:

```console
poetry run python baler --project=firstProject --mode=decompress
```

which outputs decompressed.pickle . To double check the file sizes, we can run

```console
poetry run python baler --project=firstProject --mode=info
```

which will print the file sizes of the data we’re compressing, the compressed dataset & the decompressed dataset.

#### Plotting ####

Plotting works as before, with a minor caveat. This caveat is that the column names are currently manually implemented because I couldn’t find a simple way to store the column names (there is a good explanation for this), so it will not run immediately on the UN dataset without modifications to the config file. Plotting would look something like this however:

```console
poetry run python baler --project=firstProject --mode=plot
```

# Running with Docker #

## Prerequisites ##

  * You must have Docker installed. See this [guide](https://docs.docker.com/engine/install/ "Docker Install guide")
  * You must have carried out appropriate post installation steps. For example, for Linux systems, see this [guide](https://docs.docker.com/engine/install/linux-postinstall/ "Docker postinstall guide")

## Running ##

Running with Docker requires slight modifications to the above commands. The base command becomes:

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest 
```

Where:
  * `docker run` invokes docker and specifies the running of a container
  * `--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects` mounts the local (host) directory `./projects` to the container at `/projects`
  * `ghcr.io/uomresearchit/baler:latest` specifies the container to run
  
Therefore the three commands detailed above become:

### Train: ###

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=train
```

### Compress: ### 
```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=compress
```

### Decompress: ###
```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=decompress
```

## Running  with GPU##

Baler can be run with GPU acceleration, with will happen automatically if a GPU is available on the system.

To allow the Docker image access to the system GPU a modification to the standard command is needed. For example, to run the training command:


```console
docker run \
--gpus all 
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=train
```

Where:
  * `--gpus all` tell Docker to allow the container access to the system GPUs

## Build Docker image ##

If you would prefer not to use the Docker image hosted by the University of Manchester, you may build the image yourself. This is achived with:

```console
docker build -t baler:latest .
```

This image may be run using (e.g.):

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
baler:latest \
--project=firstProject \
--mode=train
```

## Developing using Docker image ##

Docker presents some obstacles to live development, if you wish changes to be made to a Docker container it must be rebuilt. This slows development and can be frustrating.

An alternative is to use Docker volumes (mounts between local and container file systems) to shadow the source files in the container.

An example command is given here:

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
--mount type=bind,source=${PWD}/baler/modules,target=/baler-root/baler/modules \
--mount type=bind,source=${PWD}/baler/baler.py,target=/baler-root/baler/baler.py \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=train
```

Where:
  * `--mount type=bind,source=${PWD}/baler/modules,target=/baler-root/baler/modules` mounts the local source code directory shadowing the source files built in to the container
  * `--mount type=bind,source=${PWD}/baler/baler.py,target=/baler-root/baler/baler.py` mounts the main baler source file shadowing that in the container
  
Please note, this mounting does not permanently change the behaviour of the container, for this the container must be rebuilt.

