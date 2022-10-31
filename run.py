import sys
import modules.helper as helper
import modules.models as models



def main():
    project_path, data_path, config = helper.initialize(sys.argv)
    train_set, test_set, number_of_columns = helper.process(data_path, config)

    model = models.george_SAE(n_features=number_of_columns, z_dim=25)

    test_data, reconstructed_data = helper.train(model,number_of_columns,train_set,test_set,project_path,config)
    helper.plot(test_data, reconstructed_data)

if __name__ == "__main__":
	main()
