import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
from typing import Optional, Tuple, Dict
import torch
import numpy as np


def compute_dataset_statistics(dataset) -> Dict[str, np.ndarray]:
    """
    Compute mean and std of dataset for normalization.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        Dictionary with 'mean' and 'std' arrays
    """
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for images, _ in DataLoader(dataset, batch_size=1000):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {'mean': mean.numpy(), 'std': std.numpy()}


def normalize_dataset(dataset, mean: float = None, std: float = None):
    """
    Normalize dataset using specified mean and std.
    
    Args:
        dataset: PyTorch dataset
        mean: Mean for normalization (computed from dataset if None)
        std: Std for normalization (computed from dataset if None)
    
    Returns:
        Dictionary with normalization parameters
    """
    if mean is None or std is None:
        stats = compute_dataset_statistics(dataset)
        mean = stats['mean'][0]
        std = stats['std'][0]
    
    norm_params = {
        'mean': mean,
        'std': std
    }
    
    print("=" * 60)
    print("NORMALIZATION PARAMETERS")
    print("=" * 60)
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print("=" * 60)
    
    return norm_params


def build_advanced_transforms(resize: Optional[Tuple[int, int]] = None,
                             normalize: bool = True,
                             augmentation: bool = False,
                             mean: float = 0.2860,
                             std: float = 0.3205):
    """
    Build advanced transforms with preprocessing options.
    
    Args:
        resize: Target size for resizing (height, width)
        normalize: Apply normalization
        augmentation: Apply data augmentation
        mean: Mean for normalization
        std: Std for normalization
    
    Returns:
        Composed transform
    """
    transform_list = []
    
    if resize:
        transform_list.append(transforms.Resize(resize))
    
    if augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    
    transform_list.append(transforms.ToTensor())
    
    if normalize:
        transform_list.append(transforms.Normalize((mean,), (std,)))
    
    return transforms.Compose(transform_list)


def get_class_weights(dataset) -> torch.Tensor:
    """
    Calculate class weights for weighted loss function.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        Tensor of class weights
    """
    if hasattr(dataset, 'dataset'):
        # Handle random split dataset
        labels = [dataset.dataset[i][1] for i in dataset.indices]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    
    class_weights = torch.tensor([
        total_samples / (num_classes * count) for count in class_counts.values()
    ], dtype=torch.float32)
    
    return class_weights
