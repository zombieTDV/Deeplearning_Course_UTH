import os
import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from src.models.build_model import build_resnet18, build_densenet121
from src.eval.evaluate_model import load_checkpoint
from src.data.dataloader import get_cifar10_loaders

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

@torch.no_grad()
def get_logits_and_targets(model, loader, device):
    """Extract raw logits and true targets from a DataLoader."""
    model.eval()
    all_logits = []
    all_targets = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)

def sweep_logit_bias_on_val(logits_val, targets_val, cat_idx=3, dog_idx=5, beta_range=(-2.0, 2.0), num_steps=41):
    """Grid search logit bias for cat and dog classes on validation logits."""
    betas = np.linspace(beta_range[0], beta_range[1], num_steps)
    grid_acc = np.zeros((num_steps, num_steps))
    
    best_acc = -1.0
    best_beta_cat = 0.0
    best_beta_dog = 0.0
    
    baseline_preds = np.argmax(logits_val, axis=1)
    baseline_acc = np.mean(baseline_preds == targets_val) * 100.0
    
    for i, b_cat in enumerate(betas):
        for j, b_dog in enumerate(betas):
            bias_vec = np.zeros(10)
            bias_vec[cat_idx] = b_cat
            bias_vec[dog_idx] = b_dog
            
            adjusted_logits = logits_val + bias_vec
            preds = np.argmax(adjusted_logits, axis=1)
            acc = np.mean(preds == targets_val) * 100.0
            grid_acc[i, j] = acc
            
            if acc > best_acc:
                best_acc = acc
                best_beta_cat = b_cat
                best_beta_dog = b_dog
                
    return {
        "best_val_acc": float(best_acc),
        "baseline_val_acc": float(baseline_acc),
        "best_beta_cat": float(best_beta_cat),
        "best_beta_dog": float(best_beta_dog),
        "betas": betas.tolist(),
        "grid_acc": grid_acc,
    }

def benchmark_bias_on_test(logits_test, targets_test, best_b_cat, best_b_dog, cat_idx=3, dog_idx=5):
    """Evaluate baseline vs bias-tuned predictions on test set."""
    # Baseline (beta = 0)
    preds_base = np.argmax(logits_test, axis=1)
    acc_base = np.mean(preds_base == targets_test) * 100.0
    
    # Tuned (beta = beta*)
    bias_vec = np.zeros(10)
    bias_vec[cat_idx] = best_b_cat
    bias_vec[dog_idx] = best_b_dog
    
    preds_tuned = np.argmax(logits_test + bias_vec, axis=1)
    acc_tuned = np.mean(preds_tuned == targets_test) * 100.0
    
    # Per-class accuracy comparison
    cm_base = confusion_matrix(targets_test, preds_base, labels=range(10))
    cm_tuned = confusion_matrix(targets_test, preds_tuned, labels=range(10))
    
    acc_per_class_base = (cm_base.diagonal() / cm_base.sum(axis=1) * 100.0).tolist()
    acc_per_class_tuned = (cm_tuned.diagonal() / cm_tuned.sum(axis=1) * 100.0).tolist()
    
    return {
        "test_acc_baseline": float(acc_base),
        "test_acc_tuned": float(acc_tuned),
        "acc_delta": float(acc_tuned - acc_base),
        "cat_acc_baseline": float(acc_per_class_base[cat_idx]),
        "cat_acc_tuned": float(acc_per_class_tuned[cat_idx]),
        "dog_acc_baseline": float(acc_per_class_base[dog_idx]),
        "dog_acc_tuned": float(acc_per_class_tuned[dog_idx]),
        "acc_per_class_baseline": acc_per_class_base,
        "acc_per_class_tuned": acc_per_class_tuned,
        "cm_baseline": cm_base.tolist(),
        "cm_tuned": cm_tuned.tolist(),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=0)
    
    # 1. ResNet18
    print("Extracting ResNet18 logits...")
    resnet = build_resnet18(num_classes=10, mode="finetune", device=device)
    resnet = load_checkpoint(resnet, "experiments/checkpoints/ResNet18-sota_best.pt", device)
    res_val_logits, val_targets = get_logits_and_targets(resnet, val_loader, device)
    res_test_logits, test_targets = get_logits_and_targets(resnet, test_loader, device)
    del resnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. DenseNet121
    print("Extracting DenseNet121 logits...")
    densenet = build_densenet121(num_classes=10, mode="finetune", device=device)
    densenet = load_checkpoint(densenet, "experiments/checkpoints/DenseNet121-sota_best.pt", device)
    dense_val_logits, _ = get_logits_and_targets(densenet, val_loader, device)
    dense_test_logits, _ = get_logits_and_targets(densenet, test_loader, device)
    del densenet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Soft voting ensemble logits (average logits)
    ens_val_logits = 0.5 * (res_val_logits + dense_val_logits)
    ens_test_logits = 0.5 * (res_test_logits + dense_test_logits)
    
    models = {
        "ResNet18": (res_val_logits, res_test_logits),
        "DenseNet121": (dense_val_logits, dense_test_logits),
        "Soft-Voting Ensemble": (ens_val_logits, ens_test_logits)
    }
    
    results = {}
    
    for name, (v_logits, t_logits) in models.items():
        print(f"\n=== Processing {name} ===")
        sweep_res = sweep_logit_bias_on_val(v_logits, val_targets)
        best_b_cat = sweep_res["best_beta_cat"]
        best_b_dog = sweep_res["best_beta_dog"]
        print(f"Val Baseline Acc: {sweep_res['baseline_val_acc']:.2f}% -> Best Val Acc: {sweep_res['best_val_acc']:.2f}%")
        print(f"Optimal beta_cat: {best_b_cat:.2f}, beta_dog: {best_b_dog:.2f}")
        
        bench_res = benchmark_bias_on_test(t_logits, test_targets, best_b_cat, best_b_dog)
        print(f"Test Baseline Acc: {bench_res['test_acc_baseline']:.2f}% -> Tuned Test Acc: {bench_res['test_acc_tuned']:.2f}% (Delta: {bench_res['acc_delta']:+.2f}%)")
        print(f"Cat Test Acc: {bench_res['cat_acc_baseline']:.2f}% -> {bench_res['cat_acc_tuned']:.2f}%")
        print(f"Dog Test Acc: {bench_res['dog_acc_baseline']:.2f}% -> {bench_res['dog_acc_tuned']:.2f}%")
        
        results[name] = {
            "val_sweep": {
                "baseline_val_acc": sweep_res["baseline_val_acc"],
                "best_val_acc": sweep_res["best_val_acc"],
                "best_beta_cat": sweep_res["best_beta_cat"],
                "best_beta_dog": sweep_res["best_beta_dog"],
            },
            "test_benchmark": bench_res
        }

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/logit_bias_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved sweep results to experiments/results/logit_bias_sweep_results.json")

if __name__ == "__main__":
    main()
