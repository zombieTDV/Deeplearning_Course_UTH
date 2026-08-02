"""
Data Pipeline Package

This package provides a modular data pipeline for computer vision tasks.
It encapsulates dataset loading, inspection, statistics computation,
transforms, and DataLoader creation.

Public API:
    - download_cifar10: Download CIFAR-10 dataset
    - load_cifar10_dataset: Load CIFAR-10 dataset
    - get_dataset_info: Get dataset metadata
    - count_images: Count images in dataset
    - check_distribution: Check class distribution
    - verify_labels: Verify label integrity
    - detect_corrupted_images: Detect corrupted images
    - check_image_shapes: Check image shape consistency
    - compute_dataset_statistics: Compute mean and std
    - save_statistics: Save statistics to file
    - load_statistics: Load statistics from file
    - get_normalization_transform: Get normalization parameters
    - get_train_transform: Get training transform pipeline
    - get_eval_transform: Get evaluation transform pipeline
    - get_cifar10_transforms: Get CIFAR-10 transforms
    - get_transform_config: Get predefined transform config
    - get_cifar10_loaders: Get train/val/test DataLoaders
    - get_single_loader: Get single DataLoader
    - load_config: Load configuration from YAML
    - get_dataset_config: Get dataset configuration
    - get_normalization_config: Get normalization configuration
    - get_dataloader_config: Get DataLoader configuration
    - get_transform_config: Get transform configuration
"""

from __future__ import annotations

# Dataset module
from data.dataset import (
    download_cifar10,
    get_dataset_info,
    load_cifar10_dataset,
)

# Inspection module
from data.inspection import (
    check_distribution,
    check_image_shapes,
    count_images,
    detect_corrupted_images,
    verify_labels,
)

# Statistics module
from data.statistics import (
    compute_dataset_statistics,
    get_normalization_transform,
    load_statistics,
    save_statistics,
)

# Transforms module
from data.transforms import (
    CustomTransform,
    get_cifar10_transforms,
    get_eval_transform,
    get_train_transform,
    get_transform_config,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CIFAR10_MEAN,
    CIFAR10_STD,
)

# DataLoader module
from data.dataloader import (
    get_cifar10_loaders,
    get_single_loader,
)

# Configuration module
from data.config import (
    load_config,
    get_dataset_config,
    get_normalization_config,
    get_dataloader_config,
    get_transform_config as get_transforms_config,
)

__all__ = [
    # Dataset
    "download_cifar10",
    "load_cifar10_dataset",
    "get_dataset_info",
    # Inspection
    "count_images",
    "check_distribution",
    "verify_labels",
    "detect_corrupted_images",
    "check_image_shapes",
    # Statistics
    "compute_dataset_statistics",
    "save_statistics",
    "load_statistics",
    "get_normalization_transform",
    # Transforms
    "get_train_transform",
    "get_eval_transform",
    "get_cifar10_transforms",
    "get_transform_config",
    "CustomTransform",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    # DataLoader
    "get_cifar10_loaders",
    "get_single_loader",
    # Configuration
    "load_config",
    "get_dataset_config",
    "get_normalization_config",
    "get_dataloader_config",
    "get_transforms_config",
]
