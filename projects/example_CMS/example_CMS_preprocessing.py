import uproot


def pre_processing(input_path, output_path):

    Branch = "Events"
    Collection = "recoGenJets_slimmedGenJets__PAT."
    Objects = "recoGenJets_slimmedGenJets__PAT.obj"
    dropped_variables = [
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fX",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fY",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fZ",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.qx3_",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.pdgId_",
        "recoGenJets_slimmedGenJets__PAT.obj.m_state.status_",
        "recoGenJets_slimmedGenJets__PAT.obj.mJetArea",
        "recoGenJets_slimmedGenJets__PAT.obj.mPileupEnergy",
        "recoGenJets_slimmedGenJets__PAT.obj.mPassNumber",
    ]

    # Load data
    tree = uproot.open(input_path)[Branch][Collection][Objects]
    # Type clearing
    names = type_clearing(tree)
    df = tree.arrays(names, library="pd")
    # Clean data
    df = df.drop(columns=dropped_variables)
    df = df.reset_index(drop=True)
    df = df.dropna()
    global cleared_column_names
    cleared_column_names = list(df)
    df.to_pickle(output_path)


def type_clearing(tt_tree):
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
            # print('Index ',i,' was of type ',Typename_list_values[i],'            # and was deleted from the file')
            del column_names[i]

    # Returns list of column names to use in load_data function
    return column_names


data_path_beofre_pre_processing = "./data/example_CMS/example_CMS.root"
output_path = "./data/example_CMS/example_CMS.pickle"

pre_processing(data_path_beofre_pre_processing, output_path)
