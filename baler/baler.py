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
    input_path, output_path, model_path, config, mode = helper.get_arguments()
    if mode == "train":
        train_set_norm, test_set_norm, number_of_columns, normalization_features, full_data = helper.process(input_path, config)
        #helper.to_pickle(full_data, output_path+'full_data_norm.pickle')

        ModelObject= helper.model_init(config=config)
        model = ModelObject(n_features=number_of_columns, z_dim=config["latent_space_size"])

        ## Pay attention to what dataset is being passed as training data
        test_data_tensor, reconstructed_data_tensor,trained_model = helper.train(model,number_of_columns,train_set_norm,test_set_norm,output_path,config)
        test_data = helper.detach(test_data_tensor)
        reconstructed_data = helper.detach(reconstructed_data_tensor)

        print("Un-normalzing...")
        start = time.time()
        test_data_renorm = helper.renormalize(test_data,normalization_features["True min"],normalization_features["Feature Range"],config)
        reconstructed_data_renorm = helper.renormalize(reconstructed_data,normalization_features["True min"],normalization_features["Feature Range"],config)
        end = time.time()
        print("Un-normalization took:",f"{(end - start) / 60:.3} minutes")
  
        #helper.to_pickle(test_data_renorm, output_path+'before.pickle')
        #helper.to_pickle(reconstructed_data_renorm, output_path+'after.pickle')
        normalization_features.to_csv(output_path+'cms_normalization_features.csv')
        helper.model_saver(trained_model,output_path+'model.pt')

    elif mode == "plot":
        helper.plot(input_path, output_path)
        helper.loss_plotter("projects/cms/output/loss_data.csv",output_path,config)

    elif mode == "compress":
        print("Compressing...")
        start = time.time()

        compressed, data_before = helper.compress(model_path=model_path, number_of_columns=config["number_of_columns"], input_path=input_path, config=config)

        # Converting back to numpyarray
        compressed = helper.detach(compressed)
        end = time.time()

        print("Compression took:",f"{(end - start) / 60:.3} minutes")

        helper.to_pickle(compressed, output_path+'compressed.pickle')
        helper.to_pickle(data_before, output_path+"data_pre_comp.pickle")

        if config["PCA"] == True: #False by default
            print('Doing PCA compression & decompression...')
            start = time.time()
            train_set, test_set, number_of_columns, normalization_features = helper.process(input_path, config)
            train_set_norm = helper.normalize(train_set,config)

            PCA_decompressed, PCA_compressed = helper.PCA_compression(train_set = train_set_norm,data = data_before,config=config)
            helper.to_pickle(PCA_compressed, output_path + 'PCA_compressed.pickle')
            helper.to_pickle(PCA_decompressed, output_path + 'PCA_decompressed.pickle')

    
    elif mode == "decompress":
        print("Decompressing...")
        start = time.time()   

        decompressed = helper.decompress(model_path=model_path, number_of_columns=config["number_of_columns"], input_path=input_path, config=config)

        # Converting back to numpyarray
        decompressed = helper.detach(decompressed)
        normalization_features = pd.read_csv(output_path + 'cms_normalization_features.csv')
 
        decompressed = helper.renormalize(decompressed,normalization_features["True min"],normalization_features["Feature Range"],config)
        end = time.time()
        print("Decompression took:",f"{(end - start) / 60:.3} minutes")

        helper.to_pickle(decompressed, output_path+'decompressed.pickle')

    elif mode == "info":
        print(" ========================== \n This is a mode for testing \n ========================== ")
        
        pre_compression = "projects/cms/output/cleandata_pre_comp.pickle"
        compressed = "projects/cms/output/compressed.pickle"
        decompressed = "projects/cms/output/decompressed.pickle"

        files = [pre_compression,compressed,decompressed]
        q = []
        for i in range(len(files)):
            q.append(os.stat(files[i]).st_size / (1024*1024))
        
        
        print("File size before compression: ",round(q[0],2),"MB")
        print("Compressed file size: ",round(q[1],2),"MB")
        print("De-compressed file size: ",round(q[2],2),"MB")
        print("Compression ratio:",round(q[0]/q[1],2))
        print(f"Compressed file is {round(q[1]/q[0],2)*100}% the size of the original")



if __name__ == "__main__":
    main()
