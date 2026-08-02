"""
test_statistics.py — Unit tests for data.statistics module.
"""

import json
import os
import tempfile
import pytest
import torch
from src.data.statistics import compute_dataset_statistics, save_statistics, load_statistics, get_normalization_transform


def test_imports():
    """Test that statistics module imports successfully."""
    from src.data import statistics
    assert statistics is not None


def test_save_and_load_statistics():
    """Test save_statistics and load_statistics functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = os.path.join(tmpdir, "test_stats.json")
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        
        save_statistics(mean, std, stats_file)
        assert os.path.exists(stats_file)
        
        loaded = load_statistics(stats_file)
        assert loaded['mean'] == mean
        assert loaded['std'] == std


def test_save_statistics_with_tensors():
    """Test save_statistics with torch.Tensor inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = os.path.join(tmpdir, "test_stats.json")
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        save_statistics(mean, std, stats_file)
        
        loaded = load_statistics(stats_file)
        # Use approximate comparison for floating point values
        assert loaded['mean'][0] == pytest.approx(0.485, rel=1e-5)
        assert loaded['mean'][1] == pytest.approx(0.456, rel=1e-5)
        assert loaded['mean'][2] == pytest.approx(0.406, rel=1e-5)
        assert loaded['std'][0] == pytest.approx(0.229, rel=1e-5)
        assert loaded['std'][1] == pytest.approx(0.224, rel=1e-5)
        assert loaded['std'][2] == pytest.approx(0.225, rel=1e-5)


def test_get_normalization_transform_with_values():
    """Test get_normalization_transform with custom values."""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    result_mean, result_std = get_normalization_transform(mean=mean, std=std)
    assert result_mean == mean
    assert result_std == std


def test_get_normalization_transform_with_file():
    """Test get_normalization_transform with stats file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = os.path.join(tmpdir, "test_stats.json")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        save_statistics(mean, std, stats_file)
        
        result_mean, result_std = get_normalization_transform(stats_file=stats_file)
        assert result_mean == mean
        assert result_std == std


def test_get_normalization_transform_error():
    """Test get_normalization_transform raises error when no parameters provided."""
    with pytest.raises(ValueError, match="Must provide either mean/std or stats_file"):
        get_normalization_transform()


def test_get_normalization_transform_with_tensors():
    """Test get_normalization_transform with tensor inputs."""
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    
    result_mean, result_std = get_normalization_transform(mean=mean, std=std)
    assert result_mean == [0.5, 0.5, 0.5]
    assert result_std == [0.5, 0.5, 0.5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
