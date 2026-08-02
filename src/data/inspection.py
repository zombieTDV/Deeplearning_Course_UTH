"""
inspection.py — Data validation and quality checks for CIFAR-10.

Usage:
    from data.inspection import count_images, check_distribution, verify_labels
    from data.inspection import print_dataset_statistics, plot_class_distributions
    from data.dataset import load_cifar10_dataset
    
    train_set = load_cifar10_dataset(train=True)
    val_set = load_cifar10_dataset(train=False)
    test_set = load_cifar10_dataset(train=False)
    
    count = count_images(train_set)
    dist = check_distribution(train_set)
    verify_labels(train_set)
    
    # Print statistics table
    print_dataset_statistics(train_set, val_set, test_set)
    
    # Plot class distributions for all splits
    plot_class_distributions(train_set, val_set, test_set)
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _get_targets_from_dataset(dataset: Dataset) -> torch.Tensor:
    """Extract targets from a dataset, handling both Dataset and Subset objects.

    Args:
        dataset: PyTorch Dataset instance (can be a Subset).

    Returns:
        Tensor of target labels.
    """
    if isinstance(dataset, Subset):
        # For Subset, extract targets using the indices
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
        return torch.tensor(targets)
    elif hasattr(dataset, 'targets'):
        # For regular datasets with targets attribute
        return torch.tensor(dataset.targets)
    else:
        # Fallback: iterate through dataset
        targets = []
        for _, label in dataset:
            targets.append(label)
        return torch.tensor(targets)


# ---------------------------------------------------------------------------
# Public API
## ---------------------------------------------------------------------------
def count_images(dataset: Dataset) -> int:
    """Count the total number of images in the dataset.

    Args:
        dataset: PyTorch Dataset instance.

    Returns:
        Total number of samples in the dataset.
    """
    count = len(dataset)
    logger.debug(f"Dataset contains {count} images")
    return count


def check_distribution(dataset: Dataset) -> dict[str, int]:
    """Check the distribution of classes in the dataset.

    Args:
        dataset: PyTorch Dataset instance with labels.

    Returns:
        Dictionary mapping class names to their counts.
    """
    logger.debug("Checking class distribution")
    targets = _get_targets_from_dataset(dataset)
    
    class_counts = Counter(targets.tolist())
    
    # Map to class names if available
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
        dist = {class_names[i]: class_counts[i] for i in sorted(class_counts.keys())}
    else:
        dist = {str(k): v for k, v in sorted(class_counts.items())}
    
    logger.debug(f"Class distribution: {dist}")
    return dist


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
    logger.debug("Verifying label integrity")
    targets = _get_targets_from_dataset(dataset)
    
    unique_labels = torch.unique(targets)
    num_classes = len(unique_labels)
    label_range = (int(unique_labels.min()), int(unique_labels.max()))
    
    # Check if labels are contiguous from 0
    expected_labels = torch.arange(num_classes)
    valid_labels = torch.equal(unique_labels.sort().values, expected_labels)
    
    result = {
        'num_classes': num_classes,
        'label_range': label_range,
        'valid_labels': valid_labels,
        'unique_labels': unique_labels.tolist(),
    }
    logger.debug(f"Label verification: {result}")
    return result


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
    logger.debug(f"Checking {num_samples} samples for corruption")
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
    
    result = {
        'checked': checked,
        'corrupted': len(corrupted_indices),
        'corrupted_indices': corrupted_indices,
    }
    if result['corrupted'] > 0:
        logger.warning(f"Found {result['corrupted']} corrupted images")
    else:
        logger.debug("No corrupted images found")
    return result


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
    logger.debug(f"Checking image shapes for {num_samples} samples")
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
    
    result = {
        'checked': checked,
        'unique_shapes': unique_shapes,
        'consistent': consistent,
    }
    logger.debug(f"Shape check result: {result}")
    return result


def print_dataset_statistics(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    class_names: list[str] | None = None,
) -> None:
    """Print a summary table of dataset statistics.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        class_names: Optional list of class names (for display purposes).
    """
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_test = len(test_dataset)
    
    # Get number of classes from the dataset
    if hasattr(train_dataset, 'classes'):
        n_classes = len(train_dataset.classes)
        if class_names is None:
            class_names = train_dataset.classes
    else:
        n_classes = 10  # CIFAR-10 default
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
    
    avg_train = n_train // n_classes
    avg_val = n_val // n_classes
    avg_test = n_test // n_classes
    
    print("\n" + "=" * 70)
    print(f"{'Dataset':<15} {'Samples':>12} {'Classes':>10} {'Avg Samples/Class':>20}")
    print("=" * 70)
    print(f"{'Train':<15} {n_train:>12,} {n_classes:>10} {avg_train:>20,}")
    print(f"{'Validation':<15} {n_val:>12,} {n_classes:>10} {avg_val:>20,}")
    print(f"{'Test':<15} {n_test:>12,} {n_classes:>10} {avg_test:>20,}")
    print("=" * 70 + "\n")


def plot_class_distributions(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    class_names: list[str] | None = None,
    figsize: tuple[int, int] = (12, 12),
) -> None:
    """Plot class distributions for train, validation, and test sets.

    Creates three vertically stacked subplots showing the class distribution
    for each dataset split.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        class_names: Optional list of class names. If not provided, uses
            dataset.classes or generic names.
        figsize: Figure size (width, height).
    """
    # Get class names
    if hasattr(train_dataset, 'classes'):
        n_classes = len(train_dataset.classes)
        if class_names is None:
            class_names = train_dataset.classes
    else:
        n_classes = 10  # CIFAR-10 default
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Count labels for each dataset
    def get_counts(dataset: Dataset) -> list[int]:
        targets = _get_targets_from_dataset(dataset)
        counter = Counter(targets.tolist())
        return [counter[i] for i in range(n_classes)]
    
    train_counts = get_counts(train_dataset)
    val_counts = get_counts(val_dataset)
    test_counts = get_counts(test_dataset)
    
    # Prepare datasets for plotting
    datasets = [
        ("Train Set", train_counts, len(train_dataset)),
        ("Validation Set", val_counts, len(val_dataset)),
        ("Test Set", test_counts, len(test_dataset)),
    ]
    
    # Create figure with three vertically stacked subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    bar_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for idx, (name, counts, total) in enumerate(datasets):
        ax = axes[idx]
        bars = ax.bar(class_names, counts, color=bar_colors,
                      edgecolor="black", linewidth=0.6)
        
        # Add value labels above each bar
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() + max(counts) * 0.02,
                    f"{count:,}", ha="center", va="bottom", fontsize=9)
        
        ax.set_ylabel("Number of samples")
        ax.set_title(f"CIFAR-10 Class Distribution — {name} ({total:,} samples)")
        ax.set_ylim(0, max(counts) * 1.12)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.show()
