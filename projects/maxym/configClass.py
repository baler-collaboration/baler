import pandas as pd
import pickle

class Configuration(object):
    def __init__(self):
        self.input_path = "data/maxym/george.pickle"
        #self.compression_ratio = 1.26
        self.number_of_columns = 24 # same
        self.latent_space_size = 15 # same

        self.epochs = 50
        self.early_stopping = False
        self.lr_scheduler = False
        self.patience = 100
        self.min_delta = 0
        self.model_name = "george_SAE" # same
        self.custom_norm = False #same
        self.l1 = True #same
        self.reg_param = 0.001 #same
        self.RHO = 0.05 # same
        self.lr = 0.001 # same
        self.batch_size = 512 # same
        self.save_as_root = True
        self.test_size = 0.15 # same

    def pre_processing(self):
        import pre_processing.iris as preProcessorClass
        self.unprocessed_path = "data/maxym/george.csv"
        df = pd.read_csv(self.unprocessed_path)
        with open(self.input_path, "wb") as handle:
            pickle.dump(df, handle)