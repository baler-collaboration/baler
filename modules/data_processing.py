from matplotlib.style import library
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import uproot 

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
        df = tree.arrays(Type_clearing(tree), library="pd")
    return df

def clean_data(df,config):
    df = df.drop(columns=config["dropped_variables"])
    df = df.dropna()
    #df.to_csv(config.pre_processed_csv_path)
    #columns = list(df.columns)
    return df

def normalize_data(df,config):
    if  config["custom_norm"] == True:
        pass
    elif config["custom_norm"] == False:
        pass
    return df
    epoch_loss = running_loss / counter
    print(f" Train Loss: {loss:.6f}")
    return epoch_loss

def split(df, test_size,random_state):
    return train_test_split(df, test_size=test_size, random_state=random_state)

def get_columns(df):
    return list(df.columns)
