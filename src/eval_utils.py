import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


def compute_confusion_matrix(model, test_loader, device, num_classes):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels, predicted):
                cm[t.long(), p.long()] += 1
    return cm


def per_class_metrics(cm, class_names):
    num_classes = cm.shape[0]
    results = {}
    for i in range(num_classes):
        tp = cm[i, i].item()
        fn = cm[i, :].sum().item() - tp
        fp = cm[:, i].sum().item() - tp
        tn = cm.sum().item() - tp - fn - fp

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        results[class_names[i]] = {'TPR': tpr, 'FPR': fpr, 'Precision': precision}
    return results


def evaluate_detailed(model, test_loader, device, class_names):
    num_classes = len(class_names)

    model.eval()
    correct = 0
    total = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels, predicted):
                cm[t.long(), p.long()] += 1

    accuracy = 100 * correct / total
    metrics = per_class_metrics(cm, class_names)

    print(f'{"="*60}')
    print(f'  Test Accuracy: {accuracy:.2f}%')
    print(f'{"="*60}')
    print(f'  {"Class":<15} {"TPR(Recall)":>10} {"FPR":>10} {"Precision":>10}')
    print(f'  {"-"*45}')
    for name in class_names:
        m = metrics[name]
        print(f'  {name:<15} {m["TPR"]:>10.4f} {m["FPR"]:>10.4f} {m["Precision"]:>10.4f}')
    print(f'{"="*60}')

    return accuracy, cm, metrics


def get_all_probas_and_labels(model, test_loader, device, num_classes):
    """Return all softmax probabilities (N, C) and true labels (N)."""
    model.eval()
    all_probas = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            all_probas.append(probas.cpu())
            all_labels.append(labels)
    return torch.cat(all_probas).numpy(), torch.cat(all_labels).numpy()


def compute_roc_auc_scores(probas, true_labels):
    """Return per-class ROC-AUC scores."""
    n_classes = probas.shape[1]
    scores = {}
    for i in range(n_classes):
        y_true = (true_labels == i).astype(int)
        scores[f'class_{i}'] = roc_auc_score(y_true, probas[:, i])
    scores['macro'] = sum(scores.values()) / n_classes
    return scores


def compute_pr_auc_scores(probas, true_labels):
    """Return per-class PR-AUC (Average Precision) scores."""
    n_classes = probas.shape[1]
    scores = {}
    for i in range(n_classes):
        y_true = (true_labels == i).astype(int)
        scores[f'class_{i}'] = average_precision_score(y_true, probas[:, i])
    scores['macro'] = sum(scores.values()) / n_classes
    return scores


def print_comparison_table(acc_mlp, acc_cnn, roc_mlp, roc_cnn, pr_mlp, pr_cnn,
                           mlp_params, cnn_params, class_names):
    print(f'{"="*75}')
    print(f'{"Metric":<25} {"MLP":>12} {"CNN":>12} {"Diff":>12}')
    print(f'{"-"*75}')
    print(f'{"Test Accuracy (%)":<25} {acc_mlp:>11.2f} {acc_cnn:>11.2f} {acc_cnn - acc_mlp:>+11.2f}')
    print(f'{"Macro ROC-AUC":<25} {roc_mlp["macro"]:>11.4f} {roc_cnn["macro"]:>11.4f} {roc_cnn["macro"] - roc_mlp["macro"]:>+11.4f}')
    print(f'{"Macro PR-AUC (AP)":<25} {pr_mlp["macro"]:>11.4f} {pr_cnn["macro"]:>11.4f} {pr_cnn["macro"] - pr_mlp["macro"]:>+11.4f}')
    print(f'{"Parameters":<25} {mlp_params:>11,} {cnn_params:>11,} {cnn_params - mlp_params:>+11,}')
    print(f'{"="*75}')
    print(f'\n{"Per-class ROC-AUC":<25} {"MLP":>12} {"CNN":>12}')
    print(f'{"-"*50}')
    for i, name in enumerate(class_names):
        print(f'{name:<25} {roc_mlp[f"class_{i}"]:>11.4f} {roc_cnn[f"class_{i}"]:>11.4f}')
    print(f'\n{"Per-class PR-AUC (AP)":<25} {"MLP":>12} {"CNN":>12}')
    print(f'{"-"*50}')
    for i, name in enumerate(class_names):
        print(f'{name:<25} {pr_mlp[f"class_{i}"]:>11.4f} {pr_cnn[f"class_{i}"]:>11.4f}')


def get_misclassified_images(model, test_loader, device, class_names, 
                            num_samples: int = 20) -> List[Tuple]:
    """
    Get misclassified images for error analysis.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run model on
        class_names: List of class names
        num_samples: Number of misclassified samples to return
    
    Returns:
        List of tuples (image, true_label, predicted_label, confidence)
    """
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(images)):
                if predicted[i] != labels[i]:
                    confidence = probas[i][predicted[i]].item()
                    misclassified.append((
                        images[i].cpu(),
                        labels[i].item(),
                        predicted[i].item(),
                        confidence
                    ))
                    
                    if len(misclassified) >= num_samples:
                        return misclassified
    
    return misclassified


def plot_misclassified_images(misclassified: List[Tuple], class_names,
                              save_path='../outputs/plots/misclassified.png'):
    """
    Plot misclassified images for error analysis.
    
    Args:
        misclassified: List of misclassified tuples from get_misclassified_images
        class_names: List of class names
        save_path: Path to save the plot
    """
    num_samples = len(misclassified)
    if num_samples == 0:
        print("No misclassified images to plot")
        return
    
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (image, true_label, pred_label, confidence) in enumerate(misclassified):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"True: {class_names[true_label]}\n"
                    f"Pred: {class_names[pred_label]}\n"
                    f"Conf: {confidence:.2f}",
                    fontsize=8, color='red')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def analyze_errors_by_class(model, test_loader, device, class_names) -> Dict[str, Dict]:
    """
    Analyze error patterns by class.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run model on
        class_names: List of class names
    
    Returns:
        Dictionary with error analysis per class
    """
    model.eval()
    errors_by_class = {name: {'total': 0, 'misclassified': 0, 
                              'confused_with': {}} for name in class_names}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for true_label, pred_label in zip(labels, predicted):
                true_name = class_names[true_label.item()]
                pred_name = class_names[pred_label.item()]
                
                errors_by_class[true_name]['total'] += 1
                
                if true_label != pred_label:
                    errors_by_class[true_name]['misclassified'] += 1
                    
                    if pred_name not in errors_by_class[true_name]['confused_with']:
                        errors_by_class[true_name]['confused_with'][pred_name] = 0
                    errors_by_class[true_name]['confused_with'][pred_name] += 1
    
    # Calculate error rates
    for name in class_names:
        total = errors_by_class[name]['total']
        misclassified = errors_by_class[name]['misclassified']
        errors_by_class[name]['error_rate'] = misclassified / total if total > 0 else 0
    
    return errors_by_class


def print_error_analysis(errors_by_class: Dict[str, Dict]):
    """
    Print error analysis results.
    
    Args:
        errors_by_class: Dictionary from analyze_errors_by_class
    """
    print(f'{"="*70}')
    print(f'{"Error Analysis by Class":^70}')
    print(f'{"="*70}')
    print(f'{"Class":<15} {"Total":>8} {"Errors":>8} {"Error Rate":>12} {"Top Confusion":>20}')
    print(f'{"-"*70}')
    
    for class_name, stats in errors_by_class.items():
        total = stats['total']
        errors = stats['misclassified']
        error_rate = stats['error_rate']
        
        # Get top confusion
        confused_with = stats['confused_with']
        if confused_with:
            top_confusion = max(confused_with.items(), key=lambda x: x[1])
            top_confusion_str = f"{top_confusion[0]} ({top_confusion[1]})"
        else:
            top_confusion_str = "N/A"
        
        print(f'{class_name:<15} {total:>8} {errors:>8} {error_rate:>11.2%} {top_confusion_str:>20}')
    
    print(f'{"="*70}')


def plot_per_class_accuracy(cm, class_names, save_path='../outputs/plots/per_class_accuracy.png'):
    """
    Plot per-class accuracy from confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm_np = cm.cpu().numpy()
    per_class_acc = []
    
    for i in range(len(class_names)):
        total = cm_np[i, :].sum()
        correct = cm_np[i, i]
        acc = correct / total if total > 0 else 0
        per_class_acc.append(acc)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_names, per_class_acc, color='steelblue', edgecolor='navy')
    
    ax.set_title('Per-Class Accuracy', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_xlabel('Class', fontsize=10)
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
               f'{acc:.2%}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    return per_class_acc
