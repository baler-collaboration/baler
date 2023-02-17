import uproot

class Configuration(object):
    def __init__(self):
        self.unprocessed_path = "data/cms/cms_data.root"
        self.input_path = "data/cms/cms_data.pickle"

        self.epochs = 5
        self.early_stopping = True
        self.lr_scheduler = False
        self.patience = 100
        self.min_delta = 0
        self.model_name = "george_SAE"
        self.custom_norm = False
        self.l1 = True
        self.reg_param = 0.001
        self.RHO = 0.05
        self.lr = 0.001
        self.batch_size = 512
        self.save_as_root = True
        self.test_size = 0.15

        self.number_of_columns = 8
        self.latent_space_size = 4

        self.cleared_col_names = ["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"]
        self.Branch = "Events"
        self.Collection = "recoGenJets_slimmedGenJets__PAT."
        self.Objects = "recoGenJets_slimmedGenJets__PAT.obj"
        self.dropped_variables = [
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fX",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fY",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fZ",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.qx3_",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.pdgId_",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.status_",
            "recoGenJets_slimmedGenJets__PAT.obj.mJetArea",
            "recoGenJets_slimmedGenJets__PAT.obj.mPileupEnergy",
            "recoGenJets_slimmedGenJets__PAT.obj.mPassNumber"
        ]
    def pre_processing(self):
        df = self.load_data()
        df = self.clean_data(df)
        df.to_pickle(self.input_path)

    def load_data(self):
        tree = uproot.open(self.unprocessed_path)[self.Branch][self.Collection][self.Objects]
        names = self.type_clearing(tree)
        df = tree.arrays(names, library="pd")
        return df

    def type_clearing(self,tt_tree):
        type_names = tt_tree.typenames()
        column_type = []
        column_names = []

        # In order to remove non integers or -floats in the TTree,
        # we separate the values and keys
        for keys in type_names:
            column_type.append(type_names[keys])
            column_names.append(keys)

        # Checks each value of the typename values to see if it isn't an int or
        # float, and then removes it
        for i in range(len(column_type)):
            if column_type[i] != "float[]" and column_type[i] != "int32_t[]":
                # print('Index ',i,' was of type ',Typename_list_values[i],'\
                # and was deleted from the file')
                del column_names[i]

        # Returns list of column names to use in load_data function
        return column_names

    def clean_data(self,df):
        df = df.drop(columns=self.dropped_variables)
        df = df.dropna()
        global cleared_column_names
        cleared_column_names = list(df)
        return df



