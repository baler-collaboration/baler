import sys
import modules.helper as helper
import modules.models as models

def main():
    input_path, output_path, model, config = helper.get_arguments()
    train_set, test_set, number_of_columns = helper.process(input_path, config)

    model = models.george_SAE(n_features=number_of_columns, z_dim=25)

    test_data, reconstructed_data = helper.train(model,number_of_columns,train_set,test_set,output_path,config)
    test_data_renorm = helper.undo_normalization(test_data,test_set,train_set,config)
    reconstructed_data_renorm = helper.undo_normalization(reconstructed_data,test_set,train_set,config)
    helper.plot(test_data_renorm, reconstructed_data_renorm)

if __name__ == "__main__":
    main()
