"""
Tests for external PyTorch Dataset support.
"""

import unittest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from baler.modules import data_processing


class SimpleDataset(Dataset):
    """A simple dataset that returns random tensors."""
    
    def __init__(self, size=1000, feature_dim=10):
        """Initialize with random data.
        
        Args:
            size (int): Number of samples
            feature_dim (int): Dimension of each sample
        """
        self.data = torch.randn(size, feature_dim)
    
    def __len__(self):
        """Return the size of the dataset."""
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """Return the item at the specified index."""
        return self.data[idx]


class TestExternalDataset(unittest.TestCase):
    """Test cases for external dataset support."""
    
    def test_load_external_dataset(self):
        """Test loading an external dataset."""
        # Create a simple dataset
        dataset = SimpleDataset(size=1000, feature_dim=10)
        
        # Load the dataset with data_processing.load_external_dataset
        train_loader, val_loader = data_processing.load_external_dataset(
            dataset=dataset,
            test_size=0.2,
            batch_size=32,
            shuffle=True,
            random_state=42,
            deterministic=True
        )
        
        # Check that the DataLoaders have the expected lengths
        self.assertEqual(len(train_loader.dataset), 800)
        self.assertEqual(len(val_loader.dataset), 200)
        
        # Check that the batch size is correct
        for batch in train_loader:
            self.assertEqual(batch.shape[0], 32)  # Batch size
            self.assertEqual(batch.shape[1], 10)  # Feature dimension
            break
        
        # Test with different parameters
        train_loader, val_loader = data_processing.load_external_dataset(
            dataset=dataset,
            test_size=0.5,
            batch_size=64,
            shuffle=False,
            deterministic=False
        )
        
        # Check that the DataLoaders have the expected lengths
        self.assertEqual(len(train_loader.dataset), 500)
        self.assertEqual(len(val_loader.dataset), 500)
    
    def test_seed_worker(self):
        """Test the seed_worker function for reproducibility."""
        # Create a simple dataset
        dataset = SimpleDataset(size=1000, feature_dim=10)
        
        # Create two DataLoaders with the same seed
        train_loader1, _ = data_processing.load_external_dataset(
            dataset=dataset,
            test_size=0.2,
            batch_size=32,
            shuffle=True,
            random_state=42,
            deterministic=True
        )
        
        train_loader2, _ = data_processing.load_external_dataset(
            dataset=dataset,
            test_size=0.2,
            batch_size=32,
            shuffle=True,
            random_state=42,
            deterministic=True
        )
        
        # Check that the batches are identical due to the same seed
        for batch1, batch2 in zip(train_loader1, train_loader2):
            torch.testing.assert_close(batch1, batch2)
            break


if __name__ == "__main__":
    unittest.main() 