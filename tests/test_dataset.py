"""
test_dataset.py — Unit tests for data.dataset module.
"""

import pytest
from src.data.dataset import download_cifar10, load_cifar10_dataset, get_dataset_info


def test_imports():
    """Test that dataset module imports successfully."""
    from src.data import dataset
    assert dataset is not None


def test_get_dataset_info():
    """Test get_dataset_info function."""
    # Create a mock dataset
    class MockDataset:
        def __init__(self):
            self.classes = ['class0', 'class1', 'class2']
        
        def __len__(self):
            return 100
    
    mock_ds = MockDataset()
    MockDataset.__name__ = 'MockDataset'
    
    info = get_dataset_info(mock_ds)
    
    assert info['name'] == 'MockDataset'
    assert info['num_samples'] == 100
    assert info['num_classes'] == 3
    assert info['classes'] == ['class0', 'class1', 'class2']


def test_get_dataset_info_without_classes():
    """Test get_dataset_info with dataset without classes attribute."""
    class MockDataset:
        def __len__(self):
            return 50
    
    mock_ds = MockDataset()
    mock_ds.__class__.__name__ = 'SimpleDataset'
    
    info = get_dataset_info(mock_ds)
    
    assert info['name'] == 'SimpleDataset'
    assert info['num_samples'] == 50
    assert info['classes'] == ['unknown']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
