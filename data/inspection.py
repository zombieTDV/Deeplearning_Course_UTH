"""
inspection.py — Data validation and quality checks for CIFAR-10.

Usage:
    from data.inspection import count_images, check_distribution, verify_labels
    from data.dataset import load_cifar10_dataset
    
    train_set = load_cifar10_dataset(train=True)
    count = count_images(train_set)
    dist = check_distribution(train_set)
    verify_labels(train_set)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def count_images(dataset: Dataset) -> int:
    """Count the total number of images in the dataset.

    Args:
        dataset: PyTorch Dataset instance.

    Returns:
        Total number of samples in the dataset.
    """
    return len(dataset)


def check_distribution(dataset: Dataset) -> dict[str, int]:
    """Check the distribution of classes in the dataset.

    Args:
        dataset: PyTorch Dataset instance with labels.

    Returns:
        Dictionary mapping class names to their counts.
    """
    if not hasattr(dataset, 'targets'):
        # If targets not directly available, extract them
        targets = []
        for _, label in dataset:
            targets.append(label)
        targets = torch.tensor(targets)
    else:
        targets = torch.tensor(dataset.targets)
    
    class_counts = Counter(targets.tolist())
    
    # Map to class names if available
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
        return {class_names[i]: class_counts[i] for i in sorted(class_counts.keys())}
    
    return {str(k): v for k, v in sorted(class_counts.items())}


def verify_labels(dataset: Dataset) -> dict[str, Any]:
    """Verify label integrity in the dataset.

    Args:
        dataset: PyTorch Dataset instance.

    Returns:
        Dictionary with verification results including:
            - num_classes: Number of unique labels
            - label_range: Min and max label values
            - valid_labels: Boolean indicating if all labels are valid
    """
    if not hasattr(dataset, 'targets'):
        targets = []
        for _, label in dataset:
            targets.append(label)
        targets = torch.tensor(targets)
    else:
        targets = torch.tensor(dataset.targets)
    
    unique_labels = torch.unique(targets)
    num_classes = len(unique_labels)
    label_range = (int(unique_labels.min()), int(unique_labels.max()))
    
    # Check if labels are contiguous from 0
    expected_labels = torch.arange(num_classes)
    valid_labels = torch.equal(unique_labels.sort().values, expected_labels)
    
    return {
        'num_classes': num_classes,
        'label_range': label_range,
        'valid_labels': valid_labels,
        'unique_labels': unique_labels.tolist(),
    }


def detect_corrupted_images(dataset: Dataset, num_samples: int = 100) -> dict[str, Any]:
    """Detect potentially corrupted images by checking for loading errors.

    Args:
        dataset: PyTorch Dataset instance.
        num_samples: Number of samples to check (default: 100).

    Returns:
        Dictionary with inspection results including:
            - checked: Number of samples checked
            - corrupted: Number of corrupted samples found
            - corrupted_indices: List of corrupted sample indices
    """
    corrupted_indices = []
    checked = min(num_samples, len(dataset))
    
    for i in range(checked):
        try:
            image, label = dataset[i]
            # Check if image is valid tensor
            if not isinstance(image, torch.Tensor):
                corrupted_indices.append(i)
            elif torch.isnan(image).any() or torch.isinf(image).any():
                corrupted_indices.append(i)
        except Exception:
            corrupted_indices.append(i)
    
    return {
        'checked': checked,
        'corrupted': len(corrupted_indices),
        'corrupted_indices': corrupted_indices,
    }


def check_image_shapes(dataset: Dataset, num_samples: int = 100) -> dict[str, Any]:
    """Check the shape consistency of images in the dataset.

    Args:
        dataset: PyTorch Dataset instance.
        num_samples: Number of samples to check (default: 100).

    Returns:
        Dictionary with shape information including:
            - checked: Number of samples checked
            - unique_shapes: Set of unique shapes found
            - consistent: Boolean indicating if all shapes are the same
    """
    shapes = []
    checked = min(num_samples, len(dataset))
    
    for i in range(checked):
        image, _ = dataset[i]
        if isinstance(image, torch.Tensor):
            shapes.append(tuple(image.shape))
        else:
            shapes.append(str(type(image)))
    
    unique_shapes = set(shapes)
    consistent = len(unique_shapes) == 1
    
    return {
        'checked': checked,
        'unique_shapes': unique_shapes,
        'consistent': consistent,
    }
