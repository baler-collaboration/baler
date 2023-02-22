import uproot
import pickle

class PreProcessor(object):
    def __init__(self,input_path,output_path,):
        self.input_path=input_path
        self.output_path=output_path

    def pre_processing(self):
        df = self.load_cms_data()
        df = self.preprocess_28D(df)
        print(list(df.columns))
        with open(self.output_path, "wb") as handle:
            pickle.dump(df, handle)


    def load_cms_data(self):
        """This function loads events data from open CMS root files"""
        filename = self.input_path

        # The object returned by uproot.open represents a TDirectory inside the file (/).
        # We are interested in the Events branch
        events_tree = uproot.open(filename)['Events']

        # events_tree.show(name_width=100, typename_width=100)

        # The Collection we want is: recoPFJets_ak5PFJets__RECO

        recoPFJets = events_tree['recoPFJets_ak5PFJets__RECO.']['recoPFJets_ak5PFJets__RECO.obj']
        #recoPFJets.show(name_width=100, typename_width=100)

        prefix = 'recoPFJets_ak5PFJets__RECO.obj.'
        # Store the 27 variables we are interested in to a pandas dataframe - we will only use 24 or 19 of them for compression
        dataframe = recoPFJets.arrays(
            [prefix + 'pt_', prefix + 'eta_', prefix + 'phi_', prefix + 'mass_', prefix + 'vertex_.fCoordinates.fX',
            prefix + 'vertex_.fCoordinates.fY', prefix + 'vertex_.fCoordinates.fZ', prefix + 'mJetArea', prefix + 'mPileupEnergy',
            prefix + 'm_specific.mChargedHadronEnergy', prefix + 'm_specific.mNeutralHadronEnergy',
            prefix + 'm_specific.mPhotonEnergy', prefix + 'm_specific.mElectronEnergy',
            prefix + 'm_specific.mMuonEnergy', prefix + 'm_specific.mHFHadronEnergy', prefix + 'm_specific.mHFEMEnergy',
            prefix + 'm_specific.mChargedHadronMultiplicity', prefix + 'm_specific.mNeutralHadronMultiplicity',
            prefix + 'm_specific.mPhotonMultiplicity', prefix + 'm_specific.mElectronMultiplicity', prefix + 'm_specific.mMuonMultiplicity',
            prefix + 'm_specific.mHFHadronMultiplicity', prefix + 'm_specific.mHFEMMultiplicity',
            prefix + 'm_specific.mChargedEmEnergy', prefix + 'm_specific.mChargedMuEnergy', prefix + 'm_specific.mNeutralEmEnergy',
            prefix + 'm_specific.mChargedMultiplicity', prefix + 'm_specific.mNeutralMultiplicity'],       library="pd")

        prefix2 = 'ak5PFJets_'
        # Rename the column names to be shorter
        dataframe.columns = [prefix2 + 'pt_', prefix2 + 'eta_', prefix2 + 'phi_', prefix2 + 'mass_',
                            prefix2 + 'fX', prefix2 + 'fY', prefix2 + 'fZ', prefix2 + 'mJetArea', prefix2 + 'mPileupEnergy',
                            prefix2 + 'mChargedHadronEnergy', prefix2 + 'mNeutralHadronEnergy', prefix2 + 'mPhotonEnergy',
                            prefix2 + 'mElectronEnergy', prefix2 + 'mMuonEnergy', prefix2 + 'mHFHadronEnergy',
                            prefix2 + 'mHFEMEnergy', prefix2 + 'mChargedHadronMultiplicity', prefix2 + 'mNeutralHadronMultiplicity',
                            prefix2 + 'mPhotonMultiplicity', prefix2 + 'mElectronMultiplicity', prefix2 + 'mMuonMultiplicity',
                            prefix2 + 'mHFHadronMultiplicity', prefix2 + 'mHFEMMultiplicity', prefix2 + 'mChargedEmEnergy',
                            prefix2 + 'mChargedMuEnergy', prefix2 + 'mNeutralEmEnergy', prefix2 + 'mChargedMultiplicity',
                            prefix2 + 'mNeutralMultiplicity']

        #Using energy instead of mass calculated as in Åstrand Sten's studies [4]
        #print(dataframe[prefix2 + 'mass_'])
        # dataframe[prefix2 + 'energy_'] = (dataframe[prefix2 + 'phi_']**2 + dataframe[prefix2 + 'mass_']**2) * np.cosh(dataframe[prefix2 + 'eta_'])
        #print(dataframe[prefix2 + 'energy_'])
        #print(dataframe.dtypes)
        print("\nPrinting dataframe to check content:")
        dataframe.sort_values(by=[prefix2 + 'pt_'])
        #print(dataframe.head)
        dataframe.to_csv('27D_openCMS_data.csv')
        return dataframe

    def preprocess_28D(self,data_df):
        custom_norm = False
        #data_df = data_df.drop(['entry', 'subentry'], axis=1)
        data_df = data_df.sort_values(by=['ak5PFJets_pt_'])

        # drop variables that have only zero values
        data_df = data_df.drop(['ak5PFJets_fX', 'ak5PFJets_fY', 'ak5PFJets_fZ', 'ak5PFJets_mPileupEnergy'], axis=1)
        
        # drop variables because they are mostly zero - this dataset is composed mainly by jets and very few leptons (= muons and electrons)
        #if num_variables == 19:
        #    data_df = data_df.drop(['ak5PFJets_mChargedEmEnergy', 'ak5PFJets_mChargedMuEnergy', 'ak5PFJets_mMuonEnergy',
        #                           'ak5PFJets_mMuonMultiplicity', 'ak5PFJets_mElectronEnergy'], axis=1)

        # filter out jets having pT > 8 TeV and mass > 800 GeV because they are caused by noise
        data_df = data_df[data_df.ak5PFJets_pt_ < 8000]
        data_df = data_df[data_df.ak5PFJets_mass_ < 800]

        # Normalize our data using Standard Scaler from sklearn
        # scaler = StandardScaler()
        # data_df[data_df.columns] = scaler.fit_transform(data_df)
        #min_max_scaler = MinMaxScaler()

        #if not custom_norm:
        #    # Normalize all variables in the range (0, 1) using MinMax Scaler from sklearn
        #    data_df[data_df.columns] = min_max_scaler.fit_transform(data_df)
        #else:
            # Perform a custom normalization technique for the 4-momentum variables
        #    temp_df = data_df.copy()
        #    temp_df = custom_normalization(temp_df)

        #    # Normalize the rest of the variables in the range (0, 1) using MinMax Scaler from sklearn
        #    data_df[data_df.columns] = min_max_scaler.fit_transform(data_df)

         #   data_df['ak5PFJets_pt_'] = temp_df['ak5PFJets_pt_'].values
        #    data_df['ak5PFJets_phi_'] = temp_df['ak5PFJets_phi_'].values
        ##    data_df['ak5PFJets_eta_'] = temp_df['ak5PFJets_eta_'].values
         #   data_df['ak5PFJets_mass_'] = temp_df['ak5PFJets_mass_'].values

        #dropping mass from the dataframe as we have replaced it with energy
        #data_df= data_df.drop(['ak5PFJets_mass_'], axis=1)

        #print('Normalized data:')
        #print(data_df)
        # shuffling the data before splitting
        #data_df = shuffle(data_df)

        # split the data into train and test with a ratio of 15%
        #train_set, test_set = train_test_split(data_df, test_size=0.15, random_state=1)

        #print('Train data shape: ')
        #print(train_set.shape)
        #print('Test data shape: ')
        #print(test_set.shape)

        # save preprocessed data to a csv file
        data_df.to_csv('28D_openCMS_preprocessed_data.csv')

        return data_df