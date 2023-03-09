
def set_config(c):
    c.input_path          = "data/example_LHCb/example_LHCb.pickle"
    c.compression_ratio   = 2.0
    c.epochs              = 200
    c.early_stopping      = True
    c.lr_scheduler        = False
    c.patience            = 100
    c.min_delta           = 0
    c.model_name          = "george_SAE"
    c.custom_norm         = False
    c.l1                  = True
    c.reg_param             = 0.001
    c.RHO                 = 0.05
    c.lr                  = 0.001
    c.batch_size          = 512
    c.save_as_root        = True
    c.test_size           = 0.15
