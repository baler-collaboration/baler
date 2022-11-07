import sys
import modules.helper as helper
import modules.models as models
import modules.data_processing as data_processing

def main():
    # Retreive arguments 
    input_path, output_path, model, config = helper.get_arguments()

    # Prepare data for training
    train_set, test_set, number_of_columns = helper.process(input_path, config)

    # Define the model to be used
    model = models.george_SAE(n_features=number_of_columns, z_dim=25)

    # Train
    test_data, reconstructed_data = helper.train(model,number_of_columns,train_set,test_set,output_path,config)
    test_data_renorm = helper.undo_normalization(test_data,test_set,train_set,config)
    reconstructed_data_renorm = helper.undo_normalization(reconstructed_data,test_set,train_set,config)

    # Plot
    helper.plot(test_data_renorm, reconstructed_data_renorm)

if __name__ == "__main__":
    main()
