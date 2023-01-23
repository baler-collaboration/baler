# Getting started #

Baler is currently packaged using [Poetry](https://python-poetry.org/ "Poetry"), a package manager for Python which helps with dependancy management. Installing and running using Poetry requires slight modifications to the usual Python commands, detailed [below](#installing-baler-dependancies-using-poetry).

Additionally, a Docker container is available which allows the use of Baler without worrying about dependancies or environment. Instructions for this usage are given [later in this README](#running-with-docker "Running with Docker").

## Example data ##

If you wish to use this README as a tutorial, you can use the following command to aquire example data, compabible with Baler and the configuration provided in this repository.

```console
wget http://opendata.cern.ch/record/21856/files/assets/cms/mc/RunIIFall15MiniAODv2/ZprimeToTT_M-3000_W-30_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v1/10000/DAA238E5-29D6-E511-AE59-001E67DBE3EF.root -O projects/data/cms_data.root
```

## Running locally  ##

### Installing Baler dependancies using Poetry ###

First, if you do not already have Poetry installed on your system, aquire it in a manner appropriate for your system. For example, using pip:

```console
pip install poetry
```

Now, Poetry can be used to install the project dependancies. 

In the directory in which you have cloned this repository:

```console
poetry install
```

### Running Baler ###

Baler can be run locally using a virtual environment created by Poetry. This is achived using the `poetry run` command.

#### Training ####

What --mode=train now does is that it trains a given model (where the model name is currently defined in the config file). The training is the same as before, and the model parameters (officially called the state dictionary) is now saved together. Can be ran via:

```console
poetry run python baler --config=projects/cms/configs/cms.json --input=projects/cms/data/cms_data.root --output=projects/cms/output/ --mode=train
```

which will, most importantly, output: current_model.pt . This contains all necessary model parameters.

#### Compressing ####

To do something with this model, you can now choose --mode=compress, which can be ran as

```console
poetry run python baler --config=projects/cms/configs/cms.json --input=projects/cms/data/cms_data.root --output=projects/cms/output/ --model=projects/cms/output/current_model.pt --mode=compress
```

This will output a compressed file called compressed.pickle, and this is the latent space representation of the input dataset. It will also output cleandata_pre_comp.pickle which is just the exact data being compressed.

#### Decompressing ####

To decompress the compressed file, we choose --mode=decompress and run:

```console
poetry run python baler --config=projects/cms/configs/cms.json --input=projects/cms/output/compressed.pickle --output=projects/cms/output/ --model=projects/cms/output/current_model.pt --mode=decompress
```

which outputs decompressed.pickle . To double check the file sizes, we can run

```console
poetry run python baler --config=projects/cms/configs/cms.json --input=projects/cms/output/ --output=projects/cms/output/ --mode=info
```

which will print the file sizes of the data we’re compressing, the compressed dataset & the decompressed dataset.

#### Plotting ####

Plotting works as before, with a minor caveat. This caveat is that the column names are currently manually implemented because I couldn’t find a simple way to store the column names (there is a good explanation for this), so it will not run immediately on the UN dataset without modifications to the config file. Plotting would look something like this however:

```console
poetry run python baler --config=projects/cms/configs/cms.json --input=projects/cms/output/cleandata_pre_comp.pickle --output=projects/cms/output/decompressed.pickle --mode=plot
```

# Running with Docker #

## Prerequisites ##

  * You must have Docker installed. See this [guide](https://docs.docker.com/engine/install/ "Docker Install guide")
  * You must have carried out appropriate post installation steps. For example, for Linux systems, see this [guide](https://docs.docker.com/engine/install/linux-postinstall/ "Docker postinstall guide")

## Running ##

Running with Docker requires slight modifications to the above commands. The base command becomes:

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/projects \
ghcr.io/uomresearchit/baler:latest 
```

Where:
  * `docker run` invokes docker and specifies the running of a container
  * `--mount type=bind,source=${PWD}/projects/,target=/projects` mounts the local (host) directory `./projects` to the container at `/projects`
  * `ghcr.io/uomresearchit/baler:latest` specifies the container to run
  
Therefore the three commands detailed above become:

### Train: ###

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/projects \
ghcr.io/uomresearchit/baler:latest \
--config=/projects/cms/configs/cms.json \
--input=/projects/cms/data/cms_data.root \
--output=/projects/cms/output/ \
--mode=train
```

### Compress: ### 
```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/projects \
ghcr.io/uomresearchit/baler:latest \
--config=/projects/cms/configs/cms.json \
--input=/projects/cms/data/cms_data.root \
--output=/projects/cms/output/ \
--model=/projects/cms/output/current_model.pt \
--mode=compress
```

### Decompress: ###
```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/projects \
ghcr.io/uomresearchit/baler:latest \
--config=/projects/cms/configs/cms.json \
--input=/projects/cms/output/compressed.pickle \
--output=/projects/cms/output/ \
--model=/projects/cms/output/current_model.pt \
--mode=decompress
```

## Build Docker image ##

If you would prefer not to use the Docker image produced by the University of Manchester, you may build the image ourself. This is achived with:

```console
docker build -t baler:latest .
```

This image may be run using (e.g.):

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/projects \
baler:latest \
--config=/projects/cms/configs/cms.json \
--input=/projects/cms/data/cms_data.root \
--output=/projects/cms/output/ \
--mode=train
```

## Developing using Docker image ##

Docker presents some obstacles to live development, if you wish changes to be made to a Docker container it must be rebuilt. This slows development and can be frustrating.

An alternative is to use Docker volumes (mounts between local and container file systems) to shaddow the source files in the container.

An example command is given here:

```console
docker run \
--mount type=bind,source=${PWD}/projects/,target=/projects \
--mount type=bind,source=${PWD}/baler/modules,target=/baler-root/baler/modules \
--mount type=bind,source=${PWD}/baler/baler.py,target=/baler-root/baler/baler.py \
ghcr.io/uomresearchit/baler:latest \
--config=/projects/cms/configs/cms.json \
--input=/projects/cms/data/cms_data.root \
--output=/projects/cms/output/ \
--mode=train
```

Where:
  * `--mount type=bind,source=${PWD}/baler/modules,target=/baler-root/baler/modules` mounts the local source code directory shaddowing the source files built in to the container
  * `--mount type=bind,source=${PWD}/baler/baler.py,target=/baler-root/baler/baler.py` mounts the main baler source file shadding that in the container
  
Please note, this mounting does not permentantly change the behaviour of the container, for this the contatiner must be rebuilt.

