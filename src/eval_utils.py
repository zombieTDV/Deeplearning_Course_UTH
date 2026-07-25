import torch
from sklearn.metrics import roc_auc_score, average_precision_score


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
