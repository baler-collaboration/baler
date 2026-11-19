# Using External PyTorch Datasets with Baler

Baler now supports using external PyTorch Dataset objects directly. This feature allows you to:

- Use custom dataset implementations
- Utilize pre-existing PyTorch datasets
- Work with datasets that don't easily fit into NumPy arrays
- Apply custom transformations to your data during loading

## Requirements

To use this feature, you need:

1. A class that implements the PyTorch `torch.utils.data.Dataset` interface
2. Understanding of how your dataset's dimensions map to model inputs

## Using External Datasets

### Option 1: Providing a Dataset Instance

You can directly provide a PyTorch Dataset instance to Baler by adding it to your config:

```python
import torch
from torch.utils.data import Dataset
from baler.modules.helper import Config

# Example custom dataset
class MyCustomDataset(Dataset):
    def __init__(self, data_path):
        # Load your data here
        self.data = ...  # Your data loading logic
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Return a tensor representing your data item
        return self.data[idx]

# Create your dataset
my_dataset = MyCustomDataset("path/to/data")

# Create a Baler config
config = Config(...)  # Your other config parameters

# Assign the dataset to the config
config.external_dataset = my_dataset
```

### Option 2: Providing a Dataset Class Path

You can also specify the module path to your Dataset class:

```python
# Create a Baler config
config = Config(...)  # Your other config parameters

# Specify the path to your dataset class
config.external_dataset = "mymodule.mydataset.MyCustomDataset"

# Optionally provide arguments for dataset initialization
config.dataset_args = {
    "data_path": "path/to/data",
    "transform": None
}
```

## Important Considerations

1. **Dataset Format**: Your dataset's `__getitem__` method should return tensors that match the expected input format of your chosen model architecture.

2. **Dimensions**: Make sure to set `config.data_dimension` correctly (1 or 2) based on your data.

3. **Model Type**: Set `config.model_type` to either "dense" or "convolutional" based on the architecture you want to use.

4. **Batch Structure**: 
   - For 1D data: Each item should be a 1D tensor of features
   - For 2D data with dense models: Each item should be a 2D tensor
   - For 2D data with convolutional models: Each item should be a 2D tensor (will be reshaped to include channels)

## Example

Here's a complete example using the MNIST dataset:

```python
import torch
from torchvision import datasets, transforms
from baler.modules.helper import Config

# Create MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Create a Baler config
config = Config(
    input_path="",  # Not used with external dataset
    compression_ratio=0.1,
    epochs=10,
    early_stopping=True,
    early_stopping_patience=5,
    lr_scheduler=True,
    lr_scheduler_patience=2,
    min_delta=0.001,
    model_name="Dense_AE",
    model_type="dense",
    custom_norm=False,
    l1=True,
    reg_param=0.001,
    RHO=0.05,
    lr=0.001,
    batch_size=64,
    test_size=0.2,
    data_dimension=2,
    intermittent_model_saving=False,
    separate_model_saving=False,
    intermittent_saving_patience=10,
    mse_avg=False,
    mse_sum=True,
    emd=False,
    deterministic_algorithm=True,
    apply_normalization=False  # Dataset already normalized by transform
)

# Assign the dataset
config.external_dataset = mnist_dataset
```

## Limitations

1. Currently, the compression and decompression phases still require NumPy array inputs. External datasets are only supported for the training phase.

2. Your dataset items must be directly usable by the model without additional preprocessing (beyond what your Dataset class already does).

3. If your dataset returns tuples (e.g., data and labels), you'll need to create a wrapper that only returns the data portion. 