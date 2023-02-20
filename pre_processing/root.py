import uproot

class PreProcessor(object):
    def __init__(self,input_path,output_path,):
        self.input_path=input_path
        self.output_path=output_path

    def pre_processing(self):
        df = self.load_data()
        df = self.clean_data(df)
        df.to_pickle(self.output_path)

    def load_data(self):
        tree = uproot.open(self.input_path)[self.Branch][self.Collection][self.Objects]
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