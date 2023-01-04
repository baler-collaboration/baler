import sys
import modules.helper as helper
import modules.models as models
import torch 
import torch.optim as optim
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    config, mode, project = helper.get_arguments()
    project_path = f"projects/{project}/"

    if mode == "newProject":
        helper.createNewProject(project)
    elif mode == "train":

        train_set, test_set, number_of_columns, normalization_features = helper.process(config["input_path"], config)
        train_set_norm = helper.normalize(train_set,config)
        test_set_norm = helper.normalize(test_set,config)

        ModelObject= helper.model_init(config=config)
        model = ModelObject(n_features=number_of_columns, z_dim=config["latent_space_size"])

        output_path = project_path+"training/"
        test_data_tensor, reconstructed_data_tensor = helper.train(model,number_of_columns,train_set_norm,test_set_norm,output_path,config)
        test_data = helper.detach(test_data_tensor)
        reconstructed_data = helper.detach(reconstructed_data_tensor)

        print("Un-normalzing...")
        start = time.time()
        test_data_renorm = helper.renormalize(test_data,normalization_features["True min"],normalization_features["Feature Range"],config)
        reconstructed_data_renorm = helper.renormalize(reconstructed_data,normalization_features["True min"],normalization_features["Feature Range"],config)
        end = time.time()
        print("Un-normalization took:",f"{(end - start) / 60:.3} minutes")
    
        helper.to_pickle(test_data_renorm, output_path+'before.pickle')
        helper.to_pickle(reconstructed_data_renorm, output_path+'after.pickle')
        normalization_features.to_csv(project_path+'model/cms_normalization_features.csv')
        helper.model_saver(model,project_path+'model/model.pt')

    elif mode == "plot":
        output_path = project_path+"plotting/"
        helper.plot(output_path,project_path+"training/before.pickle",project_path+"training/after.pickle")
        helper.loss_plotter(project_path+"training/loss_data.csv",output_path)

    elif mode == "compress":
        print("Compressing...")
        start = time.time()
        compressed, data_before = helper.compress(model_path=project_path+"model/model.pt", number_of_columns=config["number_of_columns"], input_path=config["input_path"], config=config)
        # Converting back to numpyarray
        compressed = helper.detach(compressed)
        end = time.time()

        print("Compression took:",f"{(end - start) / 60:.3} minutes")

        helper.to_pickle(compressed, project_path+'compressed_output/compressed.pickle')
        helper.to_pickle(data_before, project_path+"compressed_output/cleandata_pre_comp.pickle")
    
    elif mode == "decompress":
        print("Decompressing...")
        start = time.time()        
        decompressed = helper.decompress(model_path=project_path+"model/model.pt", number_of_columns=config["number_of_columns"], input_path=project_path+'compressed_output/compressed.pickle', config=config)

        # Converting back to numpyarray
        decompressed = helper.detach(decompressed)
        normalization_features = pd.read_csv(project_path + 'model/cms_normalization_features.csv')
 
        decompressed = helper.renormalize(decompressed,normalization_features["True min"],normalization_features["Feature Range"],config)
        end = time.time()
        print("Decompression took:",f"{(end - start) / 60:.3} minutes")


        if config["save_as_root"] == True: ## False by default
            helper.to_root(decompressed,config,project_path + 'decompressed_output/decompressed.root')
            helper.to_pickle(decompressed, project_path + 'decompressed_output/decompressed.pickle')
        else:
            helper.to_pickle(decompressed, project_path + 'decompressed_output/decompressed.pickle')


    elif mode == "info":
        print("================================== \n Information about your compression \n================================== ")
        
        pre_compression = project_path+"compressed_output/cleandata_pre_comp.pickle"
        compressed = project_path+"compressed_output/compressed.pickle"
        decompressed = project_path+"decompressed_output/decompressed.pickle"

        files = [pre_compression,compressed,decompressed]
        q = []
        for i in range(len(files)):
            q.append(os.stat(files[i]).st_size / (1024*1024))
        

        print(f"\nCompressed file is {round(q[1]/q[0],2)*100}% the size of the original\n")    
        print("File size before compression: ",round(q[0],2),"MB")
        print("Compressed file size: ",round(q[1],2),"MB")
        print("De-compressed file size: ",round(q[2],2),"MB")
        print("Compression ratio:",round(q[0]/q[1],2))
        



if __name__ == "__main__":
    main()
