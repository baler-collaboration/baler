# === Configuration options ===


def set_config(c):
    c.input_path = "workspaces/public_datasets/data/hurricane_isabel_data_small.npz"
    c.compression_ratio = 100
    # c.number_of_columns = 24
    # c.latent_space_size = 15
    c.epochs = 10000
    c.early_stopping = False
    c.early_stopping_patience = 100
    c.min_delta = 0
    c.lr_scheduler = True
    c.lr_scheduler_patience = 100
    c.model_name = "CFD_dense_AE"
    c.model_type = "dense"
    c.custom_norm = False
    c.l1 = True
    c.reg_param = 0.001
    c.RHO = 0.05
    c.lr = 0.001
    c.batch_size = 85
    c.test_size = 0
    c.data_dimension = 2
    c.apply_normalization = True
    c.extra_compression = False
    c.intermittent_model_saving = False
    c.intermittent_saving_patience = 100
    c.activation_extraction = False
    c.deterministic_algorithm = False
    c.compress_to_latent_space = False
    c.save_error_bounded_deltas = False
    c.error_bounded_requirement = 1
