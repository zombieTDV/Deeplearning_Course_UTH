"""
Data Pipeline Package

This package provides a modular data pipeline for computer vision tasks.
It encapsulates dataset loading, inspection, statistics computation,
transforms, and DataLoader creation.

Modules:
    - dataset: Dataset download, loading, and verification
    - inspection: Data validation and quality checks
    - statistics: Dataset statistics computation and management
    - transforms: Transform pipelines for data augmentation
    - dataloader: DataLoader creation with split persistence
    - config: Configuration loader for centralized settings

Usage:
    from src.data.dataset import load_cifar10_dataset
    from src.data.transforms import get_train_transform
    from src.data.dataloader import get_cifar10_loaders
"""
