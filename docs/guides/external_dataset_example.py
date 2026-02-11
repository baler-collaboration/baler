"""
Example script showing how to use an external PyTorch Dataset with Baler.
This example uses the MNIST dataset from torchvision.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Baler imports
from baler.modules.helper import Config


# Create a simple wrapper for MNIST that ensures it only returns data (not labels)
class MNISTDatasetWrapper(Dataset):
    def __init__(self, root='./data', train=True, transform=None, download=True):
        """Initialize the MNIST dataset wrapper.
        
        Args:
            root (str): Root directory for the dataset.
            train (bool): If True, use the training set, otherwise use the test set.
            transform (callable, optional): Optional transform to apply to the data.
            download (bool): If True, download the dataset if needed.
        """
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """Return the data at the specified index (without the label)."""
        data, _ = self.mnist[idx]  # Ignore the label
        # MNIST returns 1x28x28 tensors, but we need to reshape for Baler
        # For dense models, flatten the data
        return data.flatten()  # Flatten to a 1D tensor of 784 values


def main():
    """Main function to demonstrate using an external dataset with Baler."""
    # Create the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_dataset = MNISTDatasetWrapper(
        root='./data',
        transform=transform,
        download=True
    )
    
    # Create the Baler config
    # Note: We have to create this manually since we're not using the CLI
    config = Config(
        input_path="",  # Not used with external dataset
        compression_ratio=0.1,  # Compress to 10% of original size
        epochs=10,
        early_stopping=True,
        early_stoppin_patience=5,
        lr_scheduler=True,
        lr_scheduler_patience=2,
        min_delta=0.001,
        model_name="Dense_AE",  # Using a dense autoencoder for flattened images
        model_type="dense",
        custom_norm=False,
        l1=True,
        reg_param=0.001,
        RHO=0.05,
        lr=0.001,
        batch_size=64,
        test_size=0.2,  # 20% of data used for validation
        data_dimension=1,  # We flattened the images to 1D
        intermittent_model_saving=False,
        separate_model_saving=False,
        intermittent_saving_patience=10,
        mse_avg=False,
        mse_sum=True,
        emd=False,
        deterministic_algorithm=True,
        apply_normalization=False,  # Dataset already normalized by transform
        activation_extraction=False,
    )
    
    # Set the external dataset
    config.external_dataset = mnist_dataset
    
    # Create output directory structure
    project_path = "workspaces/examples/mnist_external_dataset"
    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, "output"), exist_ok=True)
    
    # Import and call the training function
    from baler import perform_training
    perform_training(os.path.join(project_path, "output"), config, verbose=True)
    
    print("Training completed successfully!")
    print(f"Model saved in {project_path}/output")


if __name__ == "__main__":
    main() 