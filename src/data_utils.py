import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter
from typing import Optional, Tuple, Dict
import torch


def get_fashionmnist_transforms(normalize: bool = True, resize: Optional[Tuple[int, int]] = None,
                              augmentation: bool = False, denoise: bool = False):
    """
    Get transforms for FashionMNIST dataset with various preprocessing options.
    
    Args:
        normalize: Apply normalization to images
        resize: Target size for resizing (height, width)
        augmentation: Apply data augmentation
        denoise: Apply denoising (requires preprocessing module)
    
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
        # FashionMNIST mean and std
        transform_list.append(transforms.Normalize((0.2860,), (0.3205,)))
    
    return transforms.Compose(transform_list)


def load_fashionmnist(transform, root='../data', val_split: float = 0.1):
    """
    Load FashionMNIST dataset with optional validation split.
    
    Args:
        transform: Transform to apply to images
        root: Root directory for dataset
        val_split: Fraction of training data to use for validation
    
    Returns:
        train_dataset, val_dataset (or None), test_dataset
    """
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )
    
    if val_split > 0:
        # Create validation split
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        return train_dataset, val_dataset, test_dataset
    else:
        return full_train_dataset, None, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size=64, val_dataset=None,
                    num_workers: int = 0, pin_memory: bool = False,
                    class_balanced: bool = False):
    """
    Create dataloaders with optional class balancing.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size for dataloaders
        val_dataset: Validation dataset (optional)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        class_balanced: Whether to use weighted sampler for class balancing
    
    Returns:
        train_loader, val_loader (or None), test_loader
    """
    if class_balanced:
        # Calculate class weights for balanced sampling
        if hasattr(train_dataset, 'dataset'):
            # Handle random split dataset
            labels = [train_dataset.dataset[i][1] for i in train_dataset.indices]
        else:
            labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        
        class_counts = Counter(labels)
        class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


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
