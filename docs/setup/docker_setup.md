# Running with Docker #
Setup instructions for windows are available [here](docker_setup_windows.md)

## Prerequisites ##

  * You must have Docker installed. See this [guide](https://docs.docker.com/engine/install/ "Docker Install guide")
  * You must have carried out appropriate post-installation steps. For example, for Linux systems, see this [guide](https://docs.docker.com/engine/install/linux-postinstall/ "Docker postinstall guide")

## Download Data and Project Directory ##
First download tutorial data and project directories.
```console
wget https://cernbox.cern.ch/remote.php/dav/public-files/9T2zAjpvL2ee9jZ/baler.zip
```
Unzip the files
```console
unzip baler.zip
```
Enter the root directory of baler
```console
cd baler
```

This process has created the following directory tree:
```console
tree
.
└── workspaces
    ├── CFD_example
    │   ├── data
    │   │   └── example_CFD.npz
    │   └── exampleCFD
    │       ├── config
    │       │   └── example_CFD_config.py
    │       └── output
    │           ├── compressed_output
    │           ├── decompressed_output
    │           ├── plotting
    │           └── training
    └── CMS_example
        ├── data
        │   └── example_CMS.npz
        └── example_CMS
            ├── config
            │   ├── example_CMS_analysis.py
            │   ├── example_CMS_config.py
            │   └── example_CMS_preprocessing.py
            └── output
                ├── compressed_output
                ├── decompressed_output
                ├── plotting
                └── training
```
For the tutorial example, we want to compress the data called `example_CFD.npz`. The configuration file for this, including the compression ratio, number of training epochs, input data path etc is defined in `workspaces/CFD_example/example_CFD/config/example_CFD_config.py`. the output of the compressed file is `workspaces/CFD_example/example_CFD/output/compressed_output/`.

## Running ##

### Training ###
Here is the command to start **training** the network on the example_CFD data:
```console
docker run \
-u ${UID}:${GID} \
--mount type=bind,source=${PWD}/workspaces/,target=/baler-root/workspaces \
pekman/baler:latest \
--project CFD_example example_CFD \
--mode train
```

In this command, the "fixed" lines are:
  * `docker run` invokes docker and specifies the running of a container
  * `-u ${UID}:${GID}` tells the container to use your username to create files
  * `--mount type=bind,source=${PWD}/workspaces/,target=/baler-root/workspaces` mounts the local (host) directory `./workspaces` to the container at `/baler-root/workspace`
  * `pekman/baler:latest` specifies the container to run

And the user defined lines are:
  * `--project CFD_example example_CFD` specifies the current "workspace" and project. Workspaces hold input data and projects. Projects hold configuration files and output.
  * `--mode train` specifies the current running mode of Baler. We start by training the network on the data

### Compress ###
To compress the data use `--mode compress`
```console
docker run \
-u ${UID}:${GID} \
--mount type=bind,source=${PWD}/workspaces/,target=/baler-root/workspaces \
pekman/baler:latest \
--project CFD_example example_CFD \
--mode compress
```

### Decompress ###
To decompress the data use `--mode decompress`
```console
docker run \
-u ${UID}:${GID} \
--mount type=bind,source=${PWD}/workspaces/,target=/baler-root/workspaces \
pekman/baler:latest \
--project CFD_example example_CFD \
--mode decompress
```

### Plotting ###
After that training, compression, and decompression you can plot the performance of the procedure by using `--mode plot`. In this tutorial example, the performance plot is found in `workspaces/CFD_example/exmaple_CFD/output/plotting/comparison.jpg`

```console
docker run \
-u ${UID}:${GID} \
--mount type=bind,source=${PWD}/workspaces/,target=/baler-root/workspaces \
pekman/baler:latest \
--project CFD_example example_CFD \
--mode plot
```

## Running with GPU ##

Baler can be run with GPU acceleration, to allow the Docker image access to the system GPU you need to add `--gpus all` right after `docker run` in the run command:

```console
docker run \
--gpus all \
-u ${UID}:${GID} \
--mount type=bind,source=${PWD}/workspaces/,target=/baler-root/workspaces \
pekman/baler:latest \
--project CFD_example example_CFD \
--mode plot
```

## Build Docker image ##

If you would prefer not to use the Docker image provided by us, you may build the image yourself. This is achieved with:

```console
docker build --rm -t myBaler:latest .
```

This image may be run using by specifying the image `myBaler:latest` instead of our `pekman/baler:latest` in the above base command.

## Developing using Docker image ##

Docker presents some obstacles to live development, if you wish changes to be made to a Docker container it must be rebuilt. This slows development and can be frustrating.

An alternative is to use Docker volumes (mounts between local and container file systems) to shadow the source files in the container.

An example command is given here:

```console
docker run \
-u ${UID}:${GID} \
--mount type=bind,source=${PWD}/workspaces/,target=/baler-root/workspaces \
--mount type=bind,source=${PWD}/baler/modules,target=/baler-root/baler/modules \
--mount type=bind,source=${PWD}/baler/baler.py,target=/baler-root/baler/baler.py \
pekman/baler:latest \
--project CFD_example example_CFD \
--mode train
```

Where:
  * `--mount type=bind,source=${PWD}/baler/modules,target=/baler-root/baler/modules` mounts the local source code directory shadowing the source files built into the container
  * `--mount type=bind,source=${PWD}/baler/baler.py,target=/baler-root/baler/baler.py` mounts the main baler source file shadowing that in the container
  
Please note, this mounting does not permanently change the behavior of the container, for this the container must be rebuilt.


## Running with Apptainer (Singularity) on a cluster ##

Docker is not available on all platforms, particularly high-performance or shared environments prefer not to use Docker due to security concerns. In these environments, Apptainer (formerly Singularity) is generally preferred and available.

In order to run Baler on a managed platform may require additional options to work with the system wide Apptainer configuration and respect good practice such as writing to appropriate storage areas, preferably not in on a shared storage space.

Create and enter workspace directory:
```console
mkdir workspace
cd workspace
```

Download and unzip the example datasets:
```console
wget https://cernbox.cern.ch/remote.php/dav/public-files/9T2zAjpvL2ee9jZ/baler.zip
unzip baler.zip
```
By default, Apptainer/singularity will write to your home area, this is not desirable on most remote environments. To control this:
```console
export APPTAINER_CACHEDIR=${PWD}
export SINGULARITY_CACHEDIR=${PWD}
```

To build an Apptainer sandbox, a container completely constrained within a specified local directory, the following command can be run:
```console
apptainer build --sandbox baler-sandbox docker://pekman/baler:latest
```

Where:
  * `apptainer build` specifies the building of an Apptainer image
  * `--sandbox baler-sandbox/` specifies the output directory for the sandboxed container
  * `docker://pekman/baler:latest` specifies that a the Baler Docker image should be targeted

Now that the sandbox has been created, we can run the container.

### Training ###

```console
apptainer run \
--no-home \
--no-mount bind-paths \
--pwd /baler-root \
--nv \
--bind ${PWD}/baler/workspaces/:/baler-root/workspaces \
baler-sandbox/ \
--project CFD_example example_CFD \
--mode train
```
Where:
  * `-no-home` specifies to not mount the user's home directory (small, networked storage on Blackett)
  * `--no-mount bind-paths` specifies to not mount the directories specified in the global Apptainer config
  * `--pwd /baler-root` sets the working directory for the container runtime 
  * `--nv` allows the use of Nvidia graphics cards

### Compressing ###
```console
apptainer run \
--no-home \
--no-mount bind-paths \
--pwd /baler-root \
--nv \
--bind ${PWD}/baler/workspaces/:/baler-root/workspaces \
baler-sandbox/ \
--project CFD_example example_CFD \
--mode compress
```

### Decompressing ###
```console
apptainer run \
--no-home \
--no-mount bind-paths \
--pwd /baler-root \
--nv \
--bind ${PWD}/baler/workspaces/:/baler-root/workspaces \
baler-sandbox/ \
--project CFD_example example_CFD \
--mode decompress
```

### Plotting ###
```console
apptainer run \
--no-home \
--no-mount bind-paths \
--pwd /baler-root \
--nv \
--bind ${PWD}/baler/workspaces/:/baler-root/workspaces \
baler-sandbox/ \
--project CFD_example example_CFD \
--mode plot
```
