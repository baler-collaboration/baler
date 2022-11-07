import sys
import modules.helper as helper
import modules.models as models
import pickle

def main():
    input_path, output_path, model, config, mode = helper.get_arguments()

    if mode == "train":
        train_set, test_set, number_of_columns = helper.process(input_path, config)
        model = models.george_SAE(n_features=number_of_columns, z_dim=25)
        test_data, reconstructed_data = helper.train(model,number_of_columns,train_set,test_set,output_path,config)
        test_data_renorm = helper.undo_normalization(test_data,test_set,train_set,config)
        reconstructed_data_renorm = helper.undo_normalization(reconstructed_data,test_set,train_set,config)
        with open(output_path+'before.pickle', 'wb') as handle:
            pickle.dump(test_data_renorm, handle)
        with open(output_path+'after.pickle', 'wb') as handle:
            pickle.dump(reconstructed_data_renorm, handle)

    elif mode == "plot":
        helper.plot(input_path, output_path)

if __name__ == "__main__":
    main()
