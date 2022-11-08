import sys
import modules.helper as helper
import modules.models as models
import torch 
import torch.optim as optim

def main():
    input_path, output_path, model, config, mode = helper.get_arguments()

    if mode == "train":
        train_set, test_set, number_of_columns = helper.process(input_path, config)
        model = models.george_SAE(n_features=number_of_columns, z_dim=25)
        test_data, reconstructed_data = helper.train(model,number_of_columns,train_set,test_set,output_path,config)
        test_data_renorm = helper.undo_normalization(test_data,test_set,train_set,config)
        reconstructed_data_renorm = helper.undo_normalization(reconstructed_data,test_set,train_set,config)

        helper.to_pickle(test_data_renorm, output_path+'before.pickle')
        helper.to_pickle(reconstructed_data_renorm, output_path+'after.pickle')

    elif mode == "plot":
        helper.plot(input_path, output_path)

    optimizer = optim.Adam(model.parameters(),lr=0.001)

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Model weight:")    
    print(model.weight)

    print("Model bias:")    
    print(model.bias)

    print("---")
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

if __name__ == "__main__":
    main()
