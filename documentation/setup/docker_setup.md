# Running with Docker #

## Prerequisites ##

  * You must have Docker installed. See this [guide](https://docs.docker.com/engine/install/ "Docker Install guide")
  * You must have carried out appropriate post-installation steps. For example, for Linux systems, see this [guide](https://docs.docker.com/engine/install/linux-postinstall/ "Docker postinstall guide")

## Running ##

Here is the base command for running with docker, but running examples comes further down:

```console
docker run \
-u ${UID}:${GID} \ 
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest 
[--mode=... project--=...]
```

Where:
  * `docker run` invokes docker and specifies the running of a container
  * `-u ${UID}:${GID}` tells the container to use your username to create files
  * `--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects` mounts the local (host) directory `./projects` to the container at `/projects`
  * `ghcr.io/uomresearchit/baler:latest` specifies the container to run
  
Therefore the three commands detailed above become:

### Train: ###

```console
docker run \
-u ${UID}:${GID} \ 
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=train
```

### Compress: ### 
```console
docker run \
-u ${UID}:${GID} \ 
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=compress
```

### Decompress: ###
```console
docker run \
-u ${UID}:${GID} \ 
--mount type=bind,source=${PWD}/projects/,target=/baler-root/projects \
--mount type=bind,source=${PWD}/data/,target=/baler-root/data \
ghcr.io/uomresearchit/baler:latest \
--project=firstProject \
--mode=decompress
```

## Running  with GPU ##

Baler can be run with GPU acceleration, with will happen automatically if a GPU is available on the system.

To allow the Docker image access to the system GPU a modification to the standard command is needed. For example, to run the training command:


```console
docker run \
--gpus all \
-u ${UID}:${GID} \ 
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
-u ${UID}:${GID} \ 
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
-u ${UID}:${GID} \ 
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
