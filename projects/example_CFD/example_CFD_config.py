def set_config(c):
    c.input_path = "data/example_CFD/example_CFD.npz"
    c.data_dimension               = 2
    c.compression_ratio            = 10.0
    c.apply_normalization          = False
    c.model_name                   = "Conv_AE"
    c.epochs                       = 25
    c.lr                           = 0.001
    c.batch_size                   = 1
    c.early_stopping               = True
    c.lr_scheduler                 = False




# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler_patience        = 50
    c.custom_norm                  = True
    c.l1                           = True
    c.reg_param                    = 0.001
    c.RHO                          = 0.05
    c.test_size                    = 0
    c.extra_compression            = False
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 100
    c.mse_avg                      = False
    c.mse_sum                      = True
    c.emd                          = False
    c.l1                           = True 