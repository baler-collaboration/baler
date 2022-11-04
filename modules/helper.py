import modules.models as models
import modules.training as training
import modules.plotting as plotting
import modules.data_processing as data_processing

def initialize(arguments):
    project_name = arguments[1]
    project_path = f"projects/{project_name}/"
    config_path = arguments[2]
    data_path = arguments[3]
    config = data_processing.import_config(config_path)
    return project_path, data_path, config

def process(data_path, config):
    df = data_processing.load_data(data_path,config)
    df = data_processing.clean_data(df,config)
    print("As a double check, the 159363rd pt value is before norm & de-norm: ",df.loc[df.index[159363],"recoGenJets_slimmedGenJets__PAT.obj.m_state.p4Polar_.fCoordinates.fPt"])
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
