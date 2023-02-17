class Configuration(object):
    def __init__(self):
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
        self.cleared_col_names = ["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"]
        self.test_size = 0.15
        self.Branch = "Events"
        self.Collection = "recoGenJets_slimmedGenJets__PAT."
        self.Objects = "recoGenJets_slimmedGenJets__PAT.obj"
        self.number_of_columns = 8
        self.latent_space_size = 4
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
        self.input_path = "data/cms/cms_data.root"