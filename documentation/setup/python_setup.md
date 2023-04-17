## Python Setup
A setup for windows is avaliable [here](documentation/setup/python_setup_windows.md)

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

## Working Example with Python

#### Create New Project ####
Start by creating a new project directory. This will create the standardized directory structure needed, and create a blank config and output directories. In this example, these will live under `./projects/firstProject/config.json`.\
```console
poetry run python baler --project=firstProject --mode=newProject
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
