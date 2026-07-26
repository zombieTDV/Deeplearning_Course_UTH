import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import List, Dict, Tuple


def explore_dataset(train_dataset, class_names, save_path='../outputs/images/sample_fashionmnist.png'):
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Num samples: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")

    fig = plt.figure(figsize=(14, 10))

    for i in range(10):
        ax = plt.subplot(4, 5, i + 1)
        img, label = train_dataset[i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(class_names[label], fontsize=9)
        ax.axis('off')

    # Class distribution
    ax2 = plt.subplot(4, 5, (11, 15))
    labels_idx = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_counts = [labels_idx.count(c) for c in range(len(class_names))]
    bars = ax2.bar(class_names, class_counts, color='skyblue', edgecolor='navy')
    ax2.set_title('Class Distribution (Train Set)', fontsize=10)
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.set_ylim(top=max(class_counts) * 1.1)
    for bar, count in zip(bars, class_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(count), ha='center', va='bottom', fontsize=7)

    # Pixel distribution
    ax3 = plt.subplot(4, 5, (16, 20))
    sample_img = train_dataset[0][0].numpy()
    ax3.hist(sample_img.flatten(), bins=50, color='gray', edgecolor='black', alpha=0.7)
    ax3.set_title(f'Pixel Distribution (sample: {class_names[train_dataset[0][1]]})', fontsize=10)
    ax3.set_xlabel('Pixel intensity')
    ax3.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    img_sample, _ = train_dataset[0]
    print(f"Image tensor shape: {img_sample.shape}")
    print(f"Image tensor dtype: {img_sample.dtype}")
    print(f"Pixel value range: [{img_sample.min().item():.4f}, {img_sample.max().item():.4f}]")
    print(f"Mean pixel value: {img_sample.mean().item():.4f}")
    print(f"Std pixel value: {img_sample.std().item():.4f}")


def plot_losses(losses_dict, save_path='../outputs/plots/training_losses.png'):
    """Plot loss curves for multiple models. losses_dict: {name: [losses]}"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (name, losses) in enumerate(losses_dict.items()):
        ax.plot(range(1, len(losses) + 1), losses, marker='o',
                color=colors[idx % len(colors)], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix',
                         save_path='../outputs/plots/confusion_matrix.png'):
    cm_np = cm.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm_np, cmap='Blues')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=12)

    thresh = cm_np.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm_np[i, j], ha='center', va='center',
                    fontsize=7, color='white' if cm_np[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(probas_dict, true_labels, class_names, n_cols=5,
                    save_path='../outputs/plots/roc_curves.png'):
    """ROC curves per class comparing multiple models.
    probas_dict: {model_name: array(N, C)}"""
    n_classes = len(class_names)
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(n_classes):
        ax = axes[i]
        y_true = (true_labels == i).astype(int)
        for idx, (name, probas) in enumerate(probas_dict.items()):
            fpr, tpr, _ = roc_curve(y_true, probas[:, i])
            auc_val = np.trapezoid(tpr, fpr)
            ax.plot(fpr, tpr, color=colors[idx % len(colors)],
                    label=f'{name} (AUC={auc_val:.3f})', linewidth=1.5)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(class_names[i], fontsize=10)
        ax.legend(fontsize=6, loc='lower right')
        ax.grid(alpha=0.3)

    for i in range(n_classes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pr_curves(probas_dict, true_labels, class_names, n_cols=5,
                   save_path='../outputs/plots/pr_curves.png'):
    """Precision-Recall curves per class comparing multiple models."""
    n_classes = len(class_names)
    n_rows = int(np.ceil(n_classes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(n_classes):
        ax = axes[i]
        y_true = (true_labels == i).astype(int)
        for idx, (name, probas) in enumerate(probas_dict.items()):
            precision, recall, _ = precision_recall_curve(y_true, probas[:, i])
            ap = np.trapezoid(precision, recall)
            ax.plot(recall, precision, color=colors[idx % len(colors)],
                    label=f'{name} (AP={ap:.3f})', linewidth=1.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(class_names[i], fontsize=10)
        ax.legend(fontsize=6, loc='lower left')
        ax.grid(alpha=0.3)

    for i in range(n_classes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_predictions(model, test_loader, class_names, device,
                     save_path='../outputs/plots/predictions.png'):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {class_names[predicted[i]]}\nTrue: {class_names[labels[i]]}",
                     fontsize=8, color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pixel_histograms(dataset, class_names, num_samples_per_class: int = 100,
                          save_path='../outputs/plots/pixel_histograms.png'):
    """
    Plot pixel value histograms for each class.
    
    Args:
        dataset: PyTorch dataset
        class_names: List of class names
        num_samples_per_class: Number of samples to analyze per class
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for class_idx in range(len(class_names)):
        # Collect pixel values for this class
        pixel_values = []
        samples_collected = 0
        
        for img, label in dataset:
            if label == class_idx:
                if isinstance(img, torch.Tensor):
                    pixel_values.extend(img.numpy().flatten())
                else:
                    pixel_values.extend(np.array(img).flatten())
                samples_collected += 1
                if samples_collected >= num_samples_per_class:
                    break
        
        # Plot histogram
        ax = axes[class_idx]
        ax.hist(pixel_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{class_names[class_idx]}', fontsize=10)
        ax.set_xlabel('Pixel Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def analyze_brightness(dataset, class_names, save_path='../outputs/plots/brightness_analysis.png'):
    """
    Analyze brightness distribution across classes.
    
    Args:
        dataset: PyTorch dataset
        class_names: List of class names
        save_path: Path to save the plot
    """
    brightness_by_class = {name: [] for name in class_names}
    
    for img, label in dataset:
        if isinstance(img, torch.Tensor):
            brightness = img.mean().item()
        else:
            brightness = np.array(img).mean()
        brightness_by_class[class_names[label]].append(brightness)
    
    # Plot boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = [brightness_by_class[name] for name in class_names]
    bp = ax.boxplot(data_to_plot, labels=class_names, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_title('Brightness Distribution by Class', fontsize=12)
    ax.set_ylabel('Mean Pixel Value (Brightness)', fontsize=10)
    ax.set_xlabel('Class', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    # Print statistics
    print("Brightness Statistics:")
    print(f"{'Class':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    for name in class_names:
        values = brightness_by_class[name]
        print(f"{name:<15} {np.mean(values):>10.4f} {np.std(values):>10.4f} "
              f"{np.min(values):>10.4f} {np.max(values):>10.4f}")


def detect_outliers(dataset, class_names, threshold: float = 3.0,
                    save_path='../outputs/plots/outlier_detection.png'):
    """
    Detect outlier images based on brightness statistics.
    
    Args:
        dataset: PyTorch dataset
        class_names: List of class names
        threshold: Z-score threshold for outlier detection
        save_path: Path to save the plot
    """
    brightness_by_class = {name: [] for name in class_names}
    indices_by_class = {name: [] for name in class_names}
    
    for idx, (img, label) in enumerate(dataset):
        if isinstance(img, torch.Tensor):
            brightness = img.mean().item()
        else:
            brightness = np.array(img).mean()
        brightness_by_class[class_names[label]].append(brightness)
        indices_by_class[class_names[label]].append(idx)
    
    outliers = []
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for class_idx, name in enumerate(class_names):
        values = brightness_by_class[name]
        mean = np.mean(values)
        std = np.std(values)
        
        # Find outliers
        z_scores = [(v - mean) / std for v in values]
        outlier_indices = [indices_by_class[name][i] for i, z in enumerate(z_scores) 
                          if abs(z) > threshold]
        
        if outlier_indices:
            outliers.extend([(idx, name, z_scores[i]) 
                           for i, idx in enumerate(indices_by_class[name]) 
                           if abs(z_scores[i]) > threshold])
        
        # Plot distribution
        ax = axes[class_idx]
        ax.hist(values, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(mean - threshold * std, color='red', linestyle='--', label='Outlier threshold')
        ax.axvline(mean + threshold * std, color='red', linestyle='--')
        ax.set_title(f'{name}\\n({len(outlier_indices)} outliers)', fontsize=10)
        ax.set_xlabel('Brightness', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"Detected {len(outliers)} outliers (z-score > {threshold}):")
    for idx, name, z_score in outliers[:20]:  # Show first 20
        print(f"  Index {idx}, Class {name}, Z-score: {z_score:.2f}")
    if len(outliers) > 20:
        print(f"  ... and {len(outliers) - 20} more")
    
    return outliers


def verify_labels(dataset, class_names, num_samples: int = 50,
                  save_path='../outputs/plots/label_verification.png'):
    """
    Visualize samples for label verification.
    
    Args:
        dataset: PyTorch dataset
        class_names: List of class names
        num_samples: Number of samples to display per class
        save_path: Path to save the plot
    """
    samples_by_class = {name: [] for name in class_names}
    
    for img, label in dataset:
        if len(samples_by_class[class_names[label]]) < num_samples:
            samples_by_class[class_names[label]].append(img)
    
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=(num_samples * 1.5, len(class_names) * 1.5))
    
    for class_idx, name in enumerate(class_names):
        for sample_idx in range(num_samples):
            if sample_idx < len(samples_by_class[name]):
                img = samples_by_class[name][sample_idx]
                if isinstance(img, torch.Tensor):
                    img = img.squeeze().numpy()
                else:
                    img = np.array(img).squeeze()
                
                if len(class_names) == 1:
                    ax = axes[sample_idx]
                else:
                    ax = axes[class_idx, sample_idx]
                
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
                if sample_idx == 0:
                    ax.set_title(name, fontsize=10, loc='left')
            else:
                if len(class_names) == 1:
                    axes[sample_idx].axis('off')
                else:
                    axes[class_idx, sample_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"Label verification: Displayed {num_samples} samples per class")
    print("Please manually verify that labels match the images")
