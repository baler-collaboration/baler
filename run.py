import sys
import modules.helper as helper
import modules.models as models
import torch
import torch.optim as optim

def main():
    project_path, data_path, config = helper.initialize(sys.argv)
    train_set, test_set, number_of_columns = helper.process(data_path, config)

    model = models.george_SAE(n_features=number_of_columns, z_dim=4)

    model1 = torch.nn.Linear(5,2)

    test_data, reconstructed_data = helper.train(model,number_of_columns,train_set,test_set,project_path,config)
    test_data_renorm = helper.undo_normalization(test_data,test_set,train_set,config)
    reconstructed_data_renorm = helper.undo_normalization(reconstructed_data,test_set,train_set,config)
    helper.plot(test_data_renorm, reconstructed_data_renorm)

    optimizer = optim.Adam(model1.parameters(),lr=0.001)

    print("Model's state_dict:")
    for param_tensor in model1.state_dict():
        print(param_tensor, "\t", model1.state_dict()[param_tensor].size())

    print("Model weight:")    
    print(model1.weight)

    print("Model bias:")    
    print(model1.bias)

    print("---")
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

if __name__ == "__main__":
    main()
