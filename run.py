import sys
import modules.helper as helper
import modules.models as models

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

if __name__ == "__main__":
    main()
