![example workflow](https://github.com/baler-compressor/baler/actions/workflows/test_and_lint.yaml/badge.svg)
![example workflow](https://github.com/baler-compressor/baler/actions/workflows/docker.yaml/badge.svg)

# Introduction
Baler is a tool used to test the feasibility of compressing different types of scientific data using machine learning-based autoencoders.

If you wish to contribute, please see [contributing](https://github.com/baler-compressor/baler/documentation/CONTRIBUTING.md)

---

The main object the user has to produce before using Baler is a configuration file. The configuration primarily contains:
* The path to the data which is going to be compressed
* The name of the autoencoder to be used. Either pre-defined or homemade 
* Pre-processing of the dataset, for example, the dropping of certain variables
* The number of epochs, batch size, learning rate, etc.

When provided a configuration file, Baler has 4 main running modes:
* Train: Baler will train a model to compress/decompress your data and save the model.
* Compress: Baler will compress the dataset using the model derived during the training mode
* Decompress: Baler will decompress the dataset using the model derived during the training mode
* Plot: Baler will plot the difference between the original dataset and the decompressed dataset

# Getting Started #

## Windows 10/11 Setup
* Install "git for windows": https://github.com/git-for-windows/git/releases/tag/v2.39.1.windows.1
  * For a 64 bit system, probably use this one: https://github.com/git-for-windows/git/releases/download/v2.39.1.windows.1/Git-2.39.1-64-bit.exe
* Go to your windows search bar and search for "powershell". right-click powerhsell and select "run as administrator"
* Enable Linux subsystem by entering this into the PowerShell and hitting enter: `Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux`
* Go to the windows store and download "Ubuntu 22.04.1 LTS"
* Once downloaded, open it. This will start Ubuntu as a "terminal". After picking a username and password, input the following commands into that terminal. You can copy the comands using ctrl+c or the button to the right of the text. But pasting it into the terminal can only be done by right-clicking anywhere in the terminal window.

Start by updating the Windows Subsystem for Linux
```console
wsl.exe --update
```
Then, synch your clock:
```console
sudo hwclock --hctosys
```
Update your Linux packages
```console
sudo apt-get update
```
Configure git to use tour windows credentials helper, this is necessary for you to authenticate yourself on GitHub.
```console
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager-core.exe"
```
Install pip3 for downloading python packages
```console
sudo apt-get install python3-pip
```
At this point, you have a working Linux environment and you can follow the next section for the Linux setup

## Linux Setup
For some Linux users, disable the KDE keyring
```console
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```
Install poetry for managing the python environment
```console
pip3 install poetry
```
Add poetry to path in your current session
```console
source ~/.profile
```
Clone this repository
```console
git clone https://github.com/baler-compressor/baler.git
```
Move into the Baler directory
```console
cd baler
```
Use Poetry to install the project dependencies
```console
poetry install
```
Download the tutorial dataset, this will take a while
```console
wget http://opendata.cern.ch/record/21856/files/assets/cms/mc/RunIIFall15MiniAODv2/ZprimeToTT_M-3000_W-30_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v1/10000/DAA238E5-29D6-E511-AE59-001E67DBE3EF.root -O data/firstProject/cms_data.root
```
Finally, verify that the download was successful
```console 
md5sum data/firstProject/cms_data.root 
> 28910642bf94e0fa9442bc804830f88b  data/firstProject/cms_data.root
```

## Working Example

#### Create New Project ####
Start by creating a new project directory. This will create the standardized directory structure needed, and create a blank config and output directories. In this example, these will live under `./projects/firstProject/config.json`.\
```console
poetry run python baler --mode=newProject --project=firstProject
```

#### Training ####
To train the autoencoder to compress your data, you run the following command. The config file `./projects/firstProject/config.json`. details the location of the path of the input data, the number of epochs, and all the other parameters.
```console
poetry run python baler --project=firstProject --mode=train
```

#### Compressing ####
To use the derived model for compression, you can now choose ``--mode=compress``, which can be run as
```console
poetry run python baler --project=firstProject --mode=compress
```
This will output a compressed file called "compressed.pickle", and this is the latent space representation of the input dataset. It will also output cleandata_pre_comp.pickle which is just the exact data being compressed.

#### Decompressing ####
To decompress the compressed file, we choose --mode=decompress and run:
```console
poetry run python baler --project=firstProject --mode=decompress
```
This will output "decompressed.pickle". To double-check the file sizes, we can run
```console
poetry run python baler --project=firstProject --mode=info
```
which will print the file sizes of the data we’re compressing, the compressed dataset & the decompressed dataset.

#### Plotting ####
To plot the difference of your variables before compression and after decompression, we can use the following command to generate a .pdf document under ``./projects/firstProject/plotting/comparison.pdf``

```console
poetry run python baler --project=firstProject --mode=plot
```

# Running with Docker #

## Prerequisites ##

  * You must have Docker installed. See this [guide](https://docs.docker.com/engine/install/ "Docker Install guide")
  * You must have carried out appropriate post-installation steps. For example, for Linux systems, see this [guide](https://docs.docker.com/engine/install/linux-postinstall/ "Docker postinstall guide")

## Running ##

Running with Docker requires slight modifications to the above commands. The base command becomes:

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest 
[--mode=... project--=...]
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
--gpus all \
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=train
```

Where:
  * `--gpus all` tell Docker to allow the container access to the system GPUs

## Build Docker image ##

If you would prefer not to use the Docker image hosted by the University of Manchester, you may build the image yourself. This is achieved with:

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
  * `--mount type=bind,source=${PWD}/baler/modules,target=/baler-root/baler/modules` mounts the local source code directory shadowing the source files built into the container
  * `--mount type=bind,source=${PWD}/baler/baler.py,target=/baler-root/baler/baler.py` mounts the main baler source file shadowing that in the container
  
Please note, this mounting does not permanently change the behavior of the container, for this the container must be rebuilt.


## Running with Apptainer (Singularity) ##

Docker is not available on all platforms, particularly high-performance or shared environments prefer not to use Docker due to security concerns. In these environments, Apptainer (formerly Singularity) is generally preferred and available. 

To run Baler using Apptainer, the base command can be modified as follows, e.g. for the training command:

```console
apptainer run \
--nv \
--bind ${PWD}/projects/:/baler-root/projects \
--bind ${PWD}/data/:/baler-root/data \
docker://ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=train
```

### Running on Blackett (UNIMAN GPU Cluster) ###

In order to run Baler on a managed platform may require additional options to be uesd to work with the system wide Apptainer configuration and respect good practice such as writing to appropriate storage areas.

An example implementation has been made on a Univerity of Manchester (UK) GPU equipped cluster, named Blackett.

#### Ensure the Container is **not** written to the home area ####

By default, Apptainer will write to your home area, this is not desirable on Blackett. To control this:

```console
cd /to/data/dir
export APPTAINER_CACHEDIR=${PWD} # ensure you are in hard disc area, not shared
```

#### Create an Apptainer sandbox ####

To build an Apptainer sandbox, a container completely constrained within a specified local directory, the following command can be run:

```console
apptainer build \
--sandbox baler-sandbox \
docker://ghcr.io/uomresearchit/baler:latest
```

Where:
  * `apptainer build` specifies the building of an Apptainer image
  * `--sandbox baler-sandbox/` specifies the output directory for the sandboxed container
  * `docker://ghcr.io/uomresearchit/baler:latest` specifies that a the Baler Docker image should be targeted

#### Run the Apptainer sandbox ####

Now that the sandbox has been created, we can run the container. 

```console
apptainer run \
--no-home \
--no-mount bind-paths \
--pwd /baler-root \
--nv \
--bind ${PWD}/baler/projects/:/baler-root/projects \
--bind ${PWD}/baler/data:/baler-root/data \
baler-sandbox/ \
--project=firstProject \
--mode=train
```
Where:
  * `-no-home` specifies to not mount the user's home directory (small, networked storage on Blackett)
  * `--no-mount bind-paths` specifies to not mount the directories specified in the global Apptainer config
  * `--pwd /baler-root` sets the working directory for the container runtime 
  * `--nv` allows the use of Nvidia graphics cards
