
# === Configuration options ===

def set_config(c):
    c.input_path                   = "workspaces/fpga_workspace/data/fpga_data_withNames.npz"
    c.data_dimension               = 1
    c.compression_ratio            = 2.0
    c.apply_normalization          = True
    c.model_name                   = "FPGA_prototype_model"
    c.model_type                   = "dense"
    c.epochs                       = 5
    c.lr                           = 0.001
    c.batch_size                   = 64
    c.early_stopping               = True
    c.lr_scheduler                 = True




# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler_patience        = 50
    c.custom_norm                  = True
    c.reg_param                    = 0.001
    c.RHO                          = 0.05
    c.test_size                    = 0
    c.number_of_columns            = 16
    c.latent_space_size            = 8
    c.extra_compression            = False
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 100
    c.mse_avg                      = False
    c.mse_sum                      = True
    c.emd                          = False
    c.l1                           = True
    c.activation_extraction        = False
    c.deterministic_algorithm      = False



# == hls4ml configuration options ==

    c.default_reuse_factor         = 1
    c.default_precision            = "ap_fixed<16,8>"
    c.Strategy                     = "latency"
    c.Part                         = "xcvu9p-flga2104-2L-e"
    c.ClockPeriod                  = 5
    c.IOType                       = "io_parallel" 
    c.InputShape                   = (1,16)
    c.ProjectName                  = "tiny_test_model"
    c.OutputDir                    = "workspaces/fpga_workspace/fpga_project/output/hls4ml"
    c.InputData                    = None
    c.OutputPredictions            = None
    c.csim                         = False
    c.synth                        = True
    c.cosim                        = False
    c.export                       = False
    
