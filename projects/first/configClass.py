from dataclasses import dataclass

@dataclass
class Configuration(object):
    input_path          : str   = "data/first/george.pickle"
    compression_ratio   : float = 2.0
    epochs              : int   = 2
    early_stopping      : bool  = True
    lr_scheduler        : bool  = False
    patience            : int   = 100
    min_delta           : int   = 0
    model_name          : str   = "george_SAE"
    custom_norm         : bool  = False
    l1                  : bool  = True
    reg_param           : float = 0.001
    RHO                 : float = 0.05
    lr                  : float = 0.001
    batch_size          : int   = 512
    save_as_root        : bool  = True
    test_size           : float = 0.15

    def pre_processing(self):
        import pre_processing.root as preProcessorClass
        self.unprocessed_path = "data/first/george.root"
        self.pre_processor = preProcessorClass.PreProcessor(self.unprocessed_path,self.input_path)
        self.pre_processor.number_of_columns = 8
        self.pre_processor.latent_space_size = 4
        self.pre_processor.cleared_col_names = ["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"]
        self.pre_processor.Branch = "Events"
        self.pre_processor.Collection = "recoGenJets_slimmedGenJets__PAT."
        self.pre_processor.Objects = "recoGenJets_slimmedGenJets__PAT.obj"
        self.pre_processor.dropped_variables = ["recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fX","recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fY","recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fZ","recoGenJets_slimmedGenJets__PAT.obj.m_state.qx3_","recoGenJets_slimmedGenJets__PAT.obj.m_state.pdgId_","recoGenJets_slimmedGenJets__PAT.obj.m_state.status_","recoGenJets_slimmedGenJets__PAT.obj.mJetArea","recoGenJets_slimmedGenJets__PAT.obj.mPileupEnergy","recoGenJets_slimmedGenJets__PAT.obj.mPassNumber"]
        self.pre_processor.pre_processing()