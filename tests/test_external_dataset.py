"""
Tests for external PyTorch Dataset support.
"""

import unittest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from baler.modules import data_processing


class SyntheticFeatureDataset(Dataset):
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


class SequentialFeatureDataset(Dataset):
    """A dataset that returns sequential values as features."""
    
    def __init__(self, size=100):
        """Initialize with sequential data.
        
        Args:
            size (int): Number of samples
        """
        self.data = torch.arange(size).unsqueeze(1).float()  # Shape (size, 1)
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return the item at the specified index."""
        return self.data[idx]


class TestExternalDataset(unittest.TestCase):
    """Test cases for external dataset support."""
    
    def test_load_external_dataset(self):
        """Test loading an external dataset."""
        # Create a simple dataset
        dataset = SyntheticFeatureDataset(size=1000, feature_dim=10)
        
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
        dataset = SyntheticFeatureDataset(size=1000, feature_dim=10)
        
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

    def test_load_external_dataset_edge_cases(self):
        """Test loading with edge case test_size values."""
        dataset = SyntheticFeatureDataset(size=100, feature_dim=5)

        # Test with test_size = 0.0
        train_loader, val_loader = data_processing.load_external_dataset(
            dataset=dataset,
            test_size=0.0,
            batch_size=10,
            deterministic=True
        )
        self.assertEqual(len(train_loader.dataset), 100)
        self.assertEqual(len(val_loader.dataset), 0)
        # Check if val_loader is empty
        self.assertEqual(len(list(val_loader)), 0)

        # Test with test_size = 1.0
        train_loader, val_loader = data_processing.load_external_dataset(
            dataset=dataset,
            test_size=1.0,
            batch_size=10,
            deterministic=True
        )
        self.assertEqual(len(train_loader.dataset), 0)
        self.assertEqual(len(val_loader.dataset), 100)
        # Check if train_loader is empty
        self.assertEqual(len(list(train_loader)), 0)

    def test_load_external_dataset_no_shuffle(self):
        """Test loading without shuffling to ensure order is preserved."""
        # Use an identifiable dataset with sequentially increasing values
        size = 100
        dataset = SequentialFeatureDataset(size=size)

        # First get the indices that train_test_split would select
        indices = list(range(size))
        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Now we can properly test that each loader preserves the order of its subset
        expected_train_data = dataset.data[train_indices]
        expected_val_data = dataset.data[val_indices]

        train_loader, val_loader = data_processing.load_external_dataset(
            dataset=dataset,
            test_size=0.2,
            batch_size=10,
            shuffle=False, # Important: No shuffling
            random_state=42,
            deterministic=True
        )

        # Collect data from loaders
        train_data_loaded = torch.cat([batch for batch in train_loader], dim=0)
        val_data_loaded = torch.cat([batch for batch in val_loader], dim=0)

        # Test that the ordering is preserved within each subset
        torch.testing.assert_close(sorted(train_data_loaded), sorted(expected_train_data))
        torch.testing.assert_close(sorted(val_data_loaded), sorted(expected_val_data))
        
        # Additionally test that original order is preserved (values appear in original ascending order)
        self.assertTrue(torch.all(train_data_loaded[:-1] <= train_data_loaded[1:]))
        self.assertTrue(torch.all(val_data_loaded[:-1] <= val_data_loaded[1:]))


class FeatureLabelDictDataset(Dataset):
    """A dataset that returns dictionaries with features and labels."""
    def __init__(self, size=100, feature_dim=5):
        self.features = torch.randn(size, feature_dim)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'labels': self.labels[idx]}


class TestComplexDatasetLoading(unittest.TestCase):
    """Test cases for datasets returning complex types."""

    def test_load_dict_dataset(self):
        """Test loading a dataset that returns dictionaries."""
        dataset = FeatureLabelDictDataset(size=100, feature_dim=5)

        train_loader, val_loader = data_processing.load_external_dataset(
            dataset=dataset,
            test_size=0.3,
            batch_size=16,
            shuffle=True,
            random_state=42,
            deterministic=True
        )

        # Check loader sizes
        self.assertEqual(len(train_loader.dataset), 70)
        self.assertEqual(len(val_loader.dataset), 30)

        # Check batch structure and content type
        for batch in train_loader:
            self.assertIsInstance(batch, dict)
            self.assertIn('features', batch)
            self.assertIn('labels', batch)
            self.assertIsInstance(batch['features'], torch.Tensor)
            self.assertIsInstance(batch['labels'], torch.Tensor)
            self.assertEqual(batch['features'].shape[0], 16) # Batch size
            self.assertEqual(batch['features'].shape[1], 5)  # Feature dim
            self.assertEqual(batch['labels'].shape[0], 16) # Batch size
            break # Only check first batch

        for batch in val_loader:
             self.assertIsInstance(batch, dict)
             self.assertIn('features', batch)
             self.assertIn('labels', batch)
             # Validation batch size might be smaller for the last batch
             self.assertLessEqual(batch['features'].shape[0], 16)
             self.assertEqual(batch['features'].shape[1], 5)
             break # Only check first batch


if __name__ == "__main__":
    unittest.main() 