import sys
import modules.helper as helper
import modules.models as models
import modules.data_processing as data_processing
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
                    prog = "baler.py",
                    description =   '''Baler is a machine learning based compression tool for big data.\n
                                    Baler has three running modes:\n
                                    \t1. Derivation: Using a configuration file and a "small" input dataset, Baler derives a machine learning model optimized to compress and decompress your data.\n
                                    \t2. Compression: Using a previously derived model and a large input dataset, Baler compresses your data and outputs a smaller compressed file.\n
                                    \t3. Decompression: Using a previously compressed file as input and a model, Baler decompresses your data into a larger file.''',
                    epilog = 'Enjoy!')
    parser.add_argument('--config', type=str, required=False, help='Path to config file')
    parser.add_argument('--model', type=str, required=False, help='Path to previously derived machinelearning model')
    parser.add_argument('--input', type=str, required=True, help='Path to input data set for compression')
    parser.add_argument('--output', type=str, required=True, help='Path of output data')
    return parser.parse_args()

def main():
    args=get_arguments()
    input_path = args.input
    output_path = args.output
    config = data_processing.import_config(args.config)
    
    #project_path, data_path, config = helper.initialize(sys.argv)
    train_set, test_set, number_of_columns = helper.process(input_path, config)

    model = models.george_SAE(n_features=number_of_columns, z_dim=25)

    test_data, reconstructed_data = helper.train(model,number_of_columns,train_set,test_set,output_path,config)
    test_data_renorm = helper.undo_normalization(test_data,test_set,train_set,config)
    reconstructed_data_renorm = helper.undo_normalization(reconstructed_data,test_set,train_set,config)
    helper.plot(test_data_renorm, reconstructed_data_renorm)

if __name__ == "__main__":
    main()
