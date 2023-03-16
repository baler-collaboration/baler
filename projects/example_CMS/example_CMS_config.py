def set_config(c):
    c.input_path = "data/example_CMS/example_CMS.npz"
    c.data_dimension = 1
    c.compression_ratio = 1.6
    c.apply_normalization = True
    c.model_name = "AE"
    c.epochs = 5
    c.lr = 0.001
    c.batch_size = 512
    c.early_stopping = True
    c.lr_scheduler = False

    # === Additional configuration options ===

    c.early_stopping_patience = 100
    c.min_delta = 0
    c.lr_scheduler_patience = 50
    c.custom_norm = False
    c.l1 = True
    c.reg_param = 0.001
    c.RHO = 0.05
    c.lr = 0.001
    c.batch_size = 512
    c.test_size = 0.15
    # c.number_of_columns            = 24
    # c.latent_space_size            = 15
    c.extra_compression = False
    c.intermittent_model_saving = False
    c.intermittent_saving_patience = 100
    c.mse_avg = False
    c.mse_sum = True
    c.emd = False
    c.l1 = True
    c.activation_extraction = True

    c.type_list = [
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "float64",
        "float64",
        "float64",
        "int",
        "int",
    ]
