# Workspace Configs #

## List of Workspaces and Models ##
- [Workspace - CERN's Computational Fluid Dyanamics (CFD)](#workspace-cern-computational-fluid-dyanamics-cfd)
    - [Running CFD on Dense Model](#running-cfd-on-dense-model)
    - [Running CFD on Dense Model (Blocked)](#running-cfd-on-dense-model-blocked)
    - [Running CFD on Dense Model (Error Bounded)](#running-cfd-on-dense-model-error-bounded)
    - [Running CFD on 2D CNN Model](#running-cfd-on-2d-cnn-model)
    - [Running CFD on 2D CNN with GDN Activation Function (Blocked)](#running-cfd-on-2d-cnn-with-gdn-activation-function-blocked)
    - [Running CFD on 2D CNN with GDN and Slized WAE Loss Function (Blocked)](#running-cfd-on-2d-cnn-with-gdn-and-slized-wae-loss-function-blocked)
    - [Running CFD on 3D CNN Model (Blocked)](#running-cfd-on-3d-cnn-model-blocked) 
- [Workspace - CERN's Computational Fluid Mechanics (CMS)](#workspace-cern-computational-fluid-mechanics-cms)
    - [Running CMS on Dense Model](#running-cms-on-dense-model)
- [Workspace - Hurricane Isabel Dataset](#workspace-hurricane-isabel-dataset)
    - [Running Hurricane on Dense Network](#running-hurricane-on-dense-network)
- [Workspace - SLAC's Exafel Dataset](#workspace-slac-exafel-dataset)

## Workspace CERN Computational Fluid Dyanamics CFD ##

Out of the Box Config -  `./workspaces/CFD_workspace/CFD_prokect_animation/config/CFD_project_animation_config`

### Running CFD on Dense Model
Config - Use out of Box Config

### Running CFD on Dense Model Blocked
If a model is blocked then the model has the additional overhead of stitching the images together. To do so, Please one or more of following -
- Increase Number of Epochs
- Increase Model Size/Decrease Compression Ratio

Recommended changes -
```
epochs = 10000
compression_ratio = 25
convert_to_blocks = [10, 10, 10]
```

### Running CFD on Dense Model Error Bounded
Currently, Only Supported for Dense Networks!
```
model_name = "CFD_dense_AE"
save_error_bounded_deltas = True
error_bounded_requirement = 10
```

### Running CFD on 2D CNN Model
Use Conv_AE model out of the box. 
```
model_name = "Conv_AE"
model_type = "convolutional"
```

### Running CFD on 2D CNN with GDN Activation Function Blocked
The GDN Activation Function is very notorious for converging to local minima instead of global minima so hyper-parameter tuning to your dataset should give good results. Refer to [Blocked comments](#running-cfd-on-dense-model-blocked).

```
epochs = 20000
model_name = "Conv_AE_GDN"
model_type = "convolutional"
compression_ratio = 50
convert_to_blocks = [5, 5, 5]

Update model Conv_AE_GDN q_z_output_dim to 1728
```

### Running CFD on 2D CNN with GDN and Slized WAE Loss Function Blocked
Use the custom loss function to use Slized WAE
```
epochs = 20000
model_name = "Conv_AE_GDN"
model_type = "convolutional"
compression_ratio = 50
convert_to_blocks = [5, 5, 5]
custom_loss_function = "loss_function_swae"

Update model Conv_AE_GDN q_z_output_dim to 128
```

### Running CFD on 3D CNN Model Blocked
TODO

## Workspace CERN Computational Fluid Mechanics CMS ##

### Running CMS On Dense Model
TODO

## Workspace Hurricane Isabel Dataset ##

### Running Hurricane on Dense Network
```
apply_normalization = True
custom_norm = False
```

## Workspace Slac Exafel Dataset ##