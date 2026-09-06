import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def explore_dataset(train_dataset, class_names, save_path='../outputs/practice_1/images/sample_fashionmnist.png'):
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


def plot_losses(losses_dict, title='Training Loss Comparison',
                save_path='../outputs/practice_1/plots/training_losses.png'):
    """Plot loss curves for multiple models. losses_dict: {name: [losses]}"""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (name, losses) in enumerate(losses_dict.items()):
        ax.plot(range(1, len(losses) + 1), losses,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix',
                         save_path='../outputs/practice_1/plots/confusion_matrix.png'):
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
                    save_path='../outputs/practice_1/plots/roc_curves.png'):
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
                   save_path='../outputs/practice_1/plots/pr_curves.png'):
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


def plot_class_distribution_comparison(train_dataset, test_dataset, class_names,
                                        save_path='../outputs/practice_1/plots/class_distribution_comparison.png'):
    fig, ax = plt.subplots(figsize=(10, 6))

    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]

    train_counts = [train_labels.count(c) for c in range(len(class_names))]
    test_counts = [test_labels.count(c) for c in range(len(class_names))]

    x = np.arange(len(class_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_counts, width, label='Train',
                   color='skyblue', edgecolor='navy')
    bars2 = ax.bar(x + width / 2, test_counts, width, label='Test',
                   color='salmon', edgecolor='darkred')

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution Comparison: Train vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    train_total = len(train_dataset)
    test_total = len(test_dataset)
    print(f'\nTrain set size: {train_total}  |  Test set size: {test_total}')
    print(f'Class proportions (Train):  {[f"{c/train_total*100:.1f}%" for c in train_counts]}')
    print(f'Class proportions (Test):   {[f"{c/test_total*100:.1f}%" for c in test_counts]}')

    import os
    metrics_dir = '../outputs/practice_1/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'class_distribution.txt'), 'w') as f:
        f.write(f'Train samples: {train_total}, Test samples: {test_total}\n\n')
        f.write(f'{"Class":<15} {"Train Count":>12} {"Train %":>10} {"Test Count":>12} {"Test %":>10}\n')
        for i, name in enumerate(class_names):
            tp = train_counts[i] / train_total * 100
            tep = test_counts[i] / test_total * 100
            f.write(f'{name:<15} {train_counts[i]:>12} {tp:>9.1f}% {test_counts[i]:>12} {tep:>9.1f}%\n')


def plot_predictions(model, test_loader, class_names, device,
                     save_path='../outputs/practice_1/plots/predictions.png'):
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
