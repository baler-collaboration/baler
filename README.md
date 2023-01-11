# Baler
## Introduction
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

## How to Run
Start by creating a new project directory. This will create the standardised directory structure needed, and create a blank config file for you under `./projects/firstProject/config.json`.\
`python3 run.py --mode=newProject --project=firstProject`

Add a dataset to the `./data/` directory. if you are just trying things out, you can download a sample dataset using:\
`wget http://opendata.cern.ch/record/6010/files/assets/cms/Run2012B/JetHT/AOD/22Jan2013-v1/20000/CACF9904-3473-E211-AE0B-002618943857.root -P ./data/firstProject/`

Open your config file under `./projects/firstProject/config.json`, and edit the follwoing lines:
* `"input_path":"./data/firstProject/CACF9904-3473-E211-AE0B-002618943857.root",`
* `"cleared_col_names":["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"],`
* `"Branch" : "Events",`
* `"Collection": "recoGenJets_slimmedGenJets__PAT.",`
* `"Objects": "recoGenJets_slimmedGenJets__PAT.obj",`
* `"number_of_columns" : 8,`
* `"latent_space_size" : 4,`
* `"dropped_variables":["recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fX","recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fY","recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fZ","recoGenJets_slimmedGenJets__PAT.obj.m_state.qx3_","recoGenJets_slimmedGenJets__PAT.obj.m_state.pdgId_","recoGenJets_slimmedGenJets__PAT.obj.m_state.status_","recoGenJets_slimmedGenJets__PAT.obj.mJetArea","recoGenJets_slimmedGenJets__PAT.obj.mPileupEnergy","recoGenJets_slimmedGenJets__PAT.obj.mPassNumber"]`

Train the network of "firstProject" by giving baler the project path and training mode as arguments:\
`python3 run.py --project=firstProject --mode=train`

Then plot the performance using:\
`python3 run.py --project=firstProject --mode=plot`

After training and plotting you can evaluate the perforamnce of the network using the plots in `./projects/firstProject/plotting/comparison.pdf`. To improve the performance you can either change some simple parameters like the number of epocs in the config, or you can edit the model used in `./modules/models.py/`. Just make sure to reference to the correct model in your config. For example: `"model_name" : "new_SAE"`.

Once you are happy with the performance of your model you are ready to compress the data. The compressed data will show up in `./projects/firstProject/compressed_output/`. To do this, use:\
`python3 run.py --project=firstProject --mode=compress`

Later, when the time comes to use your data, you can simply decompress the data. The decompressed data will show up in `./projects/firstProject/decompressed_output/`:\
`python3 run.py --project=firstProject --mode=decompress`