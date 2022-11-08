import modules.models as models
import modules.training as training
import modules.plotting as plotting
import modules.data_processing as data_processing
import argparse
import json
import pickle

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
    parser.add_argument('--mode', type=str, required=True, help='train, compress, decompress, plot')
    args = parser.parse_args()
    if args.config: config = data_processing.import_config(args.config)
    else: config = args.config
    return args.input, args.output, args.model, config, args.mode

def to_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)

def process(data_path, config):
    df = data_processing.load_data(data_path,config)
    df = data_processing.clean_data(df,config)
    df = data_processing.normalize_data(df,config)
    train_set, test_set = data_processing.split(df, test_size=config["test_size"], random_state=1)
    number_of_columns = len(data_processing.get_columns(df))
    assert number_of_columns == config["number_of_columns"], f"The number of columns of dataframe is {number_of_columns}, config states {config['number_of_columns']}."
    return train_set, test_set, number_of_columns

def train(model,number_of_columns,train_set,test_set,project_path,config):
    return training.train(model,number_of_columns,train_set,test_set,project_path,config)

#This returns a thing called data which is the final
def undo_normalization(data,test_set,train_set,config):
    return data_processing.undo_normalization(data, test_set,train_set, config)
    
def plot(test_data, reconstructed_data):
    plotting.plot(test_data, reconstructed_data)
