import sys
import modules.helper as helper
import modules.models as models
import torch 
import torch.optim as optim
import time

def main():
    input_path, output_path, model, config, mode = helper.get_arguments()

    if mode == "train":
        train_set, test_set, number_of_columns = helper.process(input_path, config)
        model = models.george_SAE(n_features=number_of_columns, z_dim=config["latent_space_size"])
        test_data, reconstructed_data, encoded_data = helper.train(model,number_of_columns,train_set,test_set,output_path,config)

        print("Un-normalzing...")
        start = time.time()
        test_data_renorm = helper.undo_normalization(test_data,test_set,train_set,config)
        reconstructed_data_renorm = helper.undo_normalization(reconstructed_data,test_set,train_set,config)
        end = time.time()
        print("Un-normalization took: ",f"{(end - start) / 60:.3} minutes")

        helper.to_pickle(encoded_data, output_path+'encoded_data.pickle')
        helper.to_pickle(test_data_renorm, output_path+'before.pickle')
        helper.to_pickle(reconstructed_data_renorm, output_path+'after.pickle')
        helper.model_saver(model,output_path+'model_george.pt')

    elif mode == "plot":
        helper.plot(input_path, output_path)
        helper.loss_plotter("projects/cms/output/loss_val_data.csv","projects/cms/output/loss_train_data.csv",output_path)

    elif model == True and mode == "compress":
        # We need to process the data first
        train_set, test_set, number_of_columns = helper.process(input_path, config)

        # Load the model
        model = helper.model_loader(model)
    
        # Find a way to use the encoder with the given model. Currently the encoder is a model

    elif mode == "info":# and model == True):
        print("\n Loading the model and printing some information \n")
        print("================================================ \n ")
        model = helper.model_loader(model)

        print("Model name and structure: \n",model.eval())
        params = list(model.parameters())
        print("Length of parameter list:",len(params))
        print("Size of each element in params:",params[0].size())        




if __name__ == "__main__":
    main()
