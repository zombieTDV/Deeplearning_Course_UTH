import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional, List, Tuple
import torch


def analyze_dataset(dataset, class_names=None):
    """
    Analyze dataset information and print summary.
    
    Args:
        dataset: PyTorch dataset
        class_names: List of class names (optional)
    
    Returns:
        Dictionary with dataset information
    """
    info = {
        'num_samples': len(dataset),
        'image_shape': dataset[0][0].shape,
        'image_dtype': str(dataset[0][0].dtype)
    }
    
    # Get labels
    if hasattr(dataset, 'dataset'):
        labels = [dataset.dataset[i][1] for i in dataset.indices]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    info['num_classes'] = len(set(labels))
    info['class_names'] = class_names
    
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    print(f"Number of samples: {info['num_samples']}")
    print(f"Number of classes: {info['num_classes']}")
    if class_names:
        print(f"Classes: {class_names}")
    print(f"Image shape: {info['image_shape']}")
    print(f"Image dtype: {info['image_dtype']}")
    print("=" * 60)
    
    return info


def compute_pixel_statistics(dataset, num_samples: int = 1000):
    """
    Compute pixel statistics (mean, std, min, max) from dataset.
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to analyze
    
    Returns:
        Dictionary with pixel statistics
    """
    pixel_values = []
    samples_collected = 0
    
    for img, _ in dataset:
        if isinstance(img, torch.Tensor):
            pixel_values.extend(img.numpy().flatten())
        else:
            pixel_values.extend(np.array(img).flatten())
        
        samples_collected += 1
        if samples_collected >= num_samples:
            break
    
    pixel_values = np.array(pixel_values)
    
    stats = {
        'mean': pixel_values.mean(),
        'std': pixel_values.std(),
        'min': pixel_values.min(),
        'max': pixel_values.max(),
        'median': np.median(pixel_values),
        'q25': np.percentile(pixel_values, 25),
        'q75': np.percentile(pixel_values, 75)
    }
    
    print("=" * 60)
    print("PIXEL STATISTICS")
    print("=" * 60)
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std: {stats['std']:.4f}")
    print(f"Min: {stats['min']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"25th Percentile: {stats['q25']:.4f}")
    print(f"75th Percentile: {stats['q75']:.4f}")
    print("=" * 60)
    
    return stats


def check_class_distribution(dataset, class_names=None, plot: bool = True, 
                            save_path='../outputs/plots/class_distribution.png'):
    """
    Check and visualize class distribution to detect imbalance.
    
    Args:
        dataset: PyTorch dataset
        class_names: List of class names (optional)
        plot: Whether to plot the distribution
        save_path: Path to save the plot
    
    Returns:
        Dictionary with class distribution
    """
    # Get labels
    if hasattr(dataset, 'dataset'):
        labels = [dataset.dataset[i][1] for i in dataset.indices]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    distribution = {
        'class_counts': dict(class_counts),
        'num_classes': num_classes,
        'total_samples': total_samples,
        'imbalance_ratio': imbalance_ratio,
        'is_balanced': imbalance_ratio < 1.5
    }
    
    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {num_classes}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    print(f"Is balanced: {distribution['is_balanced']}")
    print("-" * 60)
    
    if class_names:
        for i, name in enumerate(class_names):
            count = class_counts.get(i, 0)
            percentage = count / total_samples * 100
            print(f"{name:<20} {count:>8} ({percentage:>5.2f}%)")
    else:
        for class_idx in sorted(class_counts.keys()):
            count = class_counts[class_idx]
            percentage = count / total_samples * 100
            print(f"Class {class_idx:<15} {count:>8} ({percentage:>5.2f}%)")
    
    print("=" * 60)
    
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if class_names:
            names = class_names
            counts = [class_counts.get(i, 0) for i in range(len(class_names))]
        else:
            names = [f"Class {i}" for i in sorted(class_counts.keys())]
            counts = [class_counts[i] for i in sorted(class_counts.keys())]
        
        bars = ax.bar(names, counts, color='steelblue', edgecolor='navy')
        ax.set_title('Class Distribution', fontsize=12)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlabel('Class', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f'{count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
    
    return distribution


def visualize_random_samples(dataset, class_names=None, num_samples: int = 16,
                            save_path='../outputs/plots/random_samples.png'):
    """
    Visualize random samples from dataset.
    
    Args:
        dataset: PyTorch dataset
        class_names: List of class names (optional)
        num_samples: Number of samples to display
        save_path: Path to save the plot
    """
    import random
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        img, label = dataset[sample_idx]
        
        if isinstance(img, torch.Tensor):
            img = img.squeeze().numpy()
        else:
            img = np.array(img).squeeze()
        
        ax.imshow(img, cmap='gray')
        
        if class_names:
            title = f"{class_names[label]}"
        else:
            title = f"Class {label}"
        
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(indices), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"Displayed {len(indices)} random samples")


def detect_outliers(dataset, threshold: float = 3.0, plot: bool = True,
                    save_path='../outputs/plots/outlier_detection.png'):
    """
    Detect outlier images based on brightness statistics using z-score.
    
    Args:
        dataset: PyTorch dataset
        threshold: Z-score threshold for outlier detection
        plot: Whether to plot the distribution
        save_path: Path to save the plot
    
    Returns:
        List of (index, label, z_score) tuples for outliers
    """
    brightness_values = []
    indices = []
    labels = []
    
    for idx, (img, label) in enumerate(dataset):
        if isinstance(img, torch.Tensor):
            brightness = img.mean().item()
        else:
            brightness = np.array(img).mean()
        
        brightness_values.append(brightness)
        indices.append(idx)
        labels.append(label)
    
    brightness_values = np.array(brightness_values)
    mean_brightness = brightness_values.mean()
    std_brightness = brightness_values.std()
    
    # Calculate z-scores
    z_scores = (brightness_values - mean_brightness) / std_brightness
    
    # Find outliers
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    outliers = [(indices[i], labels[i], z_scores[i]) for i in outlier_indices]
    
    print("=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    print(f"Mean brightness: {mean_brightness:.4f}")
    print(f"Std brightness: {std_brightness:.4f}")
    print(f"Threshold (z-score): {threshold}")
    print(f"Number of outliers: {len(outliers)}")
    print("=" * 60)
    
    if outliers:
        print("Outlier samples:")
        for idx, label, z_score in outliers[:10]:
            print(f"  Index {idx}, Label {label}, Z-score: {z_score:.2f}")
        if len(outliers) > 10:
            print(f"  ... and {len(outliers) - 10} more")
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(brightness_values, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(mean_brightness - threshold * std_brightness, color='red', 
                  linestyle='--', label=f'Lower threshold ({-threshold}σ)')
        ax.axvline(mean_brightness + threshold * std_brightness, color='red', 
                  linestyle='--', label=f'Upper threshold ({threshold}σ)')
        ax.axvline(mean_brightness, color='blue', linestyle='-', label='Mean')
        ax.set_title('Brightness Distribution with Outlier Thresholds', fontsize=12)
        ax.set_xlabel('Brightness (Mean Pixel Value)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
    
    return outliers
