class Configuration(object):
    def __init__(self):
        self.input_path = "data/maxym/maxym.pickle"
        #self.compression_ratio = 1.26
        self.number_of_columns = 24
        self.latent_space_size = 15

        self.epochs = 50
        self.early_stopping = False
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

        #self.cleared_col_names = ["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"]
        #self.number_of_columns = 8
        #self.latent_space_size = 4

    def pre_processing(self):
        import pre_processing.root as preProcessorClass
        self.unprocessed_path = "data/maxym/maxym.root"
        self.pre_processor = preProcessorClass.PreProcessor(self.unprocessed_path,self.input_path)
        #self.pre_processor.number_of_columns = 8
        #self.pre_processor.latent_space_size = 4
        self.pre_processor.cleared_col_names = ['recoPFJets_ak5PFJets__RECO.obj.pt_',
                                                'recoPFJets_ak5PFJets__RECO.obj.eta_',
                                                'recoPFJets_ak5PFJets__RECO.obj.phi_',
                                                'recoPFJets_ak5PFJets__RECO.obj.mass_',
                                                'recoPFJets_ak5PFJets__RECO.obj.mJetArea',
                                                'recoPFJets_ak5PFJets__RECO.obj.mChargedHadronEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mNeutralHadronEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mPhotonEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mElectronEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mMuonEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mHFHadronEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mHFEMEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mChargedHadronMultiplicity',
                                                'recoPFJets_ak5PFJets__RECO.obj.mNeutralHadronMultiplicity',
                                                'recoPFJets_ak5PFJets__RECO.obj.mPhotonMultiplicity',
                                                'recoPFJets_ak5PFJets__RECO.obj.mElectronMultiplicity',
                                                'recoPFJets_ak5PFJets__RECO.obj.mMuonMultiplicity',
                                                'recoPFJets_ak5PFJets__RECO.obj.mHFHadronMultiplicity',
                                                'recoPFJets_ak5PFJets__RECO.obj.mHFEMMultiplicity',
                                                'recoPFJets_ak5PFJets__RECO.obj.mChargedEmEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mChargedMuEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mNeutralEmEnergy',
                                                'recoPFJets_ak5PFJets__RECO.obj.mChargedMultiplicity',
                                                'recoPFJets_ak5PFJets__RECO.obj.mNeutralMultiplicity']
        self.pre_processor.Branch = "Events"
        self.pre_processor.Collection = "recoPFJets_ak5PFJets__RECO."
        self.pre_processor.Objects = "recoPFJets_ak5PFJets__RECO.obj"
        self.pre_processor.pre_processing()