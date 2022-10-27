from sklearn.model_selection import train_test_split
import json
import pandas as pd

def import_config(config_path):
    with open (config_path) as json_config:
        config = json.load(json_config)
    return config

def load_data(data_path,config):
    df = pd.read_csv(data_path,low_memory=False)
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
