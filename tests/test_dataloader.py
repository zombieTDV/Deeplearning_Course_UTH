"""
test_dataloader.py — Unit tests for data.dataloader module.
"""

import pytest
from torch.utils.data import Dataset, TensorDataset
from data.dataloader import get_single_loader


def test_imports():
    """Test that dataloader module imports successfully."""
    from data import dataloader
    assert dataloader is not None


def test_get_single_loader():
    """Test get_single_loader function."""
    # Create a simple mock dataset
    class MockDataset(Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return idx, idx % 10
    
    dataset = MockDataset()
    loader = get_single_loader(dataset, batch_size=10, shuffle=False)
    
    assert loader is not None
    assert loader.batch_size == 10


def test_get_single_loader_shuffle():
    """Test get_single_loader with shuffle enabled."""
    class MockDataset(Dataset):
        def __len__(self):
            return 50
        
        def __getitem__(self, idx):
            return idx, idx % 5
    
    dataset = MockDataset()
    loader = get_single_loader(dataset, batch_size=10, shuffle=True)
    
    assert loader is not None
    # DataLoader shuffle parameter is internal, we just verify it doesn't error


def test_get_single_loader_custom_params():
    """Test get_single_loader with custom parameters."""
    class MockDataset(Dataset):
        def __len__(self):
            return 20
        
        def __getitem__(self, idx):
            return idx, idx
    
    dataset = MockDataset()
    loader = get_single_loader(
        dataset,
        batch_size=5,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    assert loader is not None
    assert loader.batch_size == 5


def test_get_single_loader_with_tensor_dataset():
    """Test get_single_loader with TensorDataset."""
    import torch
    
    data = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(data, labels)
    
    loader = get_single_loader(dataset, batch_size=16, shuffle=False)
    
    assert loader is not None
    assert loader.batch_size == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
