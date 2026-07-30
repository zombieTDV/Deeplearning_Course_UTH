"""
test_transforms.py — Unit tests for data.transforms module.
"""

import pytest
from data.transforms import (
    get_train_transform,
    get_eval_transform,
    get_cifar10_transforms,
    get_transform_config,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CIFAR10_MEAN,
    CIFAR10_STD,
)


def test_imports():
    """Test that transforms module imports successfully."""
    from data import transforms
    assert transforms is not None


def test_constants():
    """Test that normalization constants are defined."""
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3
    assert len(CIFAR10_MEAN) == 3
    assert len(CIFAR10_STD) == 3


def test_get_train_transform_defaults():
    """Test get_train_transform with default parameters."""
    transform = get_train_transform()
    assert transform is not None


def test_get_train_transform_custom():
    """Test get_train_transform with custom parameters."""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = get_train_transform(mean=mean, std=std, resize_size=128, augmentation=False)
    assert transform is not None


def test_get_train_transform_augmentation():
    """Test get_train_transform with augmentation enabled/disabled."""
    transform_aug = get_train_transform(augmentation=True)
    transform_no_aug = get_train_transform(augmentation=False)
    assert transform_aug is not None
    assert transform_no_aug is not None


def test_get_eval_transform_defaults():
    """Test get_eval_transform with default parameters."""
    transform = get_eval_transform()
    assert transform is not None


def test_get_eval_transform_custom():
    """Test get_eval_transform with custom parameters."""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = get_eval_transform(mean=mean, std=std, resize_size=128)
    assert transform is not None


def test_get_cifar10_transforms_imagenet():
    """Test get_cifar10_transforms with ImageNet stats."""
    train_transform, eval_transform = get_cifar10_transforms(use_cifar10_stats=False)
    assert train_transform is not None
    assert eval_transform is not None


def test_get_cifar10_transforms_cifar10():
    """Test get_cifar10_transforms with CIFAR-10 stats."""
    train_transform, eval_transform = get_cifar10_transforms(use_cifar10_stats=True)
    assert train_transform is not None
    assert eval_transform is not None


def test_get_cifar10_transforms_augmentation():
    """Test get_cifar10_transforms with augmentation."""
    train_transform, eval_transform = get_cifar10_transforms(augmentation=True)
    assert train_transform is not None
    assert eval_transform is not None


def test_get_transform_config_imagenet_standard():
    """Test get_transform_config for imagenet_standard."""
    config = get_transform_config("imagenet_standard")
    assert config['mean'] == IMAGENET_MEAN
    assert config['std'] == IMAGENET_STD
    assert config['resize_size'] == 224
    assert config['augmentation'] is False


def test_get_transform_config_imagenet_augmented():
    """Test get_transform_config for imagenet_augmented."""
    config = get_transform_config("imagenet_augmented")
    assert config['mean'] == IMAGENET_MEAN
    assert config['std'] == IMAGENET_STD
    assert config['resize_size'] == 224
    assert config['augmentation'] is True


def test_get_transform_config_cifar10_standard():
    """Test get_transform_config for cifar10_standard."""
    config = get_transform_config("cifar10_standard")
    assert config['mean'] == CIFAR10_MEAN
    assert config['std'] == CIFAR10_STD
    assert config['resize_size'] == 32
    assert config['augmentation'] is False


def test_get_transform_config_cifar10_augmented():
    """Test get_transform_config for cifar10_augmented."""
    config = get_transform_config("cifar10_augmented")
    assert config['mean'] == CIFAR10_MEAN
    assert config['std'] == CIFAR10_STD
    assert config['resize_size'] == 32
    assert config['augmentation'] is True


def test_get_transform_config_invalid():
    """Test get_transform_config with invalid config name."""
    with pytest.raises(ValueError, match="Unknown config"):
        get_transform_config("invalid_config")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
