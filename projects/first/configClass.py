class Configuration(object):
    def __init__(self):
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

        self.cleared_col_names = ["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"]
        self.number_of_columns = 8
        self.latent_space_size = 4

    def pre_processing(self):
        import pre_processing.root as preProcessorClass
        self.unprocessed_path = "data/cms/cms_data.root"
        self.pre_processor = preProcessorClass.PreProcessor(self.unprocessed_path,self.input_path)
        self.pre_processor.number_of_columns = 8
        self.pre_processor.latent_space_size = 4
        self.pre_processor.cleared_col_names = ["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"]
        self.pre_processor.Branch = "Events"
        self.pre_processor.Collection = "recoGenJets_slimmedGenJets__PAT."
        self.pre_processor.Objects = "recoGenJets_slimmedGenJets__PAT.obj"
        self.pre_processor.dropped_variables = ["recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fX","recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fY","recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fZ","recoGenJets_slimmedGenJets__PAT.obj.m_state.qx3_","recoGenJets_slimmedGenJets__PAT.obj.m_state.pdgId_","recoGenJets_slimmedGenJets__PAT.obj.m_state.status_","recoGenJets_slimmedGenJets__PAT.obj.mJetArea","recoGenJets_slimmedGenJets__PAT.obj.mPileupEnergy","recoGenJets_slimmedGenJets__PAT.obj.mPassNumber"]
        self.pre_processor.pre_processing()