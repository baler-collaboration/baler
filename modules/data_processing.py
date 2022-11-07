from argparse import Namespace
from matplotlib.pyplot import sca
from matplotlib.style import library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd
import uproot 
import numpy as np

def import_config(config_path):
    with open (config_path) as json_config:
        config = json.load(json_config)
    return config

def Type_clearing(TTree):
    typenames = TTree.typenames()
    Column_Type = []
    Column_names = []

    # In order to remove non integers or -floats in the TTree, we separate the values and keys
    for keys in typenames:
        Column_Type.append(typenames[keys])
        Column_names.append(keys)
 
    # Checks each value of the typename values to see if it isn't an int or float, and then removes it
    for i in range(len(Column_Type)):
        if Column_Type[i] != 'float[]' and Column_Type[i] != 'int32_t[]':
            #print("Index ",i," was of type ",Typename_list_values[i]," and was deleted from the file")
            del Column_names[i]
    
    # Returns list of column names to use in load_data function
    return Column_names

def load_data(data_path,config):
    if ".csv" in data_path[-4:]:
        df = pd.read_csv(data_path,low_memory=False)
    elif ".root" in data_path[-5:]:
        tree = uproot.open(data_path)[config["Branch"]][config["Collection"]][config["Objects"]]
        global Names
        Names = Type_clearing(tree)
        df = tree.arrays(Names, library="pd")
    return df

def clean_data(df,config):
    df = df.drop(columns=config["dropped_variables"])
    df = df.dropna()
    global cleared_column_names
    cleared_column_names = list(df)
    #df.to_csv(config.pre_processed_csv_path)
    #columns = list(df.columns)
    return df

def normalize_data(df,config):
    if  config["custom_norm"] == True:
        pass
    elif config["custom_norm"] == False:
        global min_max_scaler
        min_max_scaler = MinMaxScaler()
        
        df = np.transpose(np.array(df))

        scaled_df = min_max_scaler.fit_transform(df)

        global scaling_array 
        scaling_array = min_max_scaler.scale_

        df = pd.DataFrame(scaled_df.T,columns=cleared_column_names)
    return df
    epoch_loss = running_loss / counter
    print(f" Train Loss: {loss:.6f}")
    return epoch_loss

def split(df, test_size,random_state):
    return train_test_split(df, test_size=test_size, random_state=random_state)

def get_columns(df):
    return list(df.columns)

def undo_normalization(data,test_set,train_set,config):
    if  config["custom_norm"] == True:
        pass
    elif config["custom_norm"] == False:

        # To find out which indices have been selected for training/testing, we check:
        scaling_list = scaling_array.tolist()
        train_set_list = train_set.index.tolist()
        test_set_list = test_set.index.tolist()

        #Get the scaling values we're interested in
        # This loop is very slow. Needs to be faster. Too tired at the moment to fix it, but I don't think its that hard.
        for index in sorted(train_set_list, reverse=True):
            del scaling_list[index]
        
        # Drop the list of variables which we won't use anymore from the dataframe:
        # We now take the trained data as input, make it a dataframe with the indices of the test_set.
        # The data will be a np.array when outputed from the training. 
        data = pd.DataFrame(data,index=test_set_list,columns=cleared_column_names)
        data = (data.T / np.array(scaling_list)).T
    return data
