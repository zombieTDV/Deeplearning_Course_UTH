# CIFAR_STEM_EXPERIMENT.md — Native 32x32 Resolution vs 224x224 Resizing Experiment

## 📌 Experiment Overview

- **Experiment Name:** Native CIFAR-10 Conv Stem Adaptation vs 224x224 Upsampling Baseline
- **Target Component:** ResNet18 & DenseNet121 Input Stem (`conv1` / `maxpool`)
- **Author:** AI Agent & Dev Team
- **Date:** 2026-08-01
- **Status:** Completed (2026-08-01)

---

## 📊 Experimental Results (Executed 2026-08-01)

The experiment benchmarked 3 epochs of training per variant on CUDA GPU.

### Comparison Matrix:

| Model Variant | Input Dim | Total Params | Trainable Params | Total Time (s) | Time/Epoch (s) | Peak VRAM (MB) | Test Loss | Test Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ResNet18_224_Frozen`** | 224x224 | 11,181,642 | 5,130 | 221.91s | 73.97s | 620.2 MB | 0.5866 | **79.76%** |
| **`ResNet18_224_Finetune`** | 224x224 | 11,181,642 | 8,398,858 | 244.22s | 81.41s | 716.3 MB | 0.2509 | **91.73%** |
| **`ResNet18_32_NativeStem_Frozen`** | 32x32 | 11,173,962 | 6,858 | 128.13s | 42.71s | 532.8 MB | 1.4648 | **48.28%** |
| **`ResNet18_32_NativeStem_Finetune`** | 32x32 | 11,173,962 | 8,400,586 | 146.80s | 48.93s | 610.4 MB | 0.7711 | **72.93%** |

---

## 📈 Benchmark Visualizations (CIFAR Stem Experiment)

Below is the collection of 7 dedicated analytical visualization plots for the CIFAR Conv Stem experiment:

1. **Master Dashboard**: [../../experiments/plots/cifar_stem_master_dashboard.png](../../experiments/plots/cifar_stem_master_dashboard.png) - 4-panel multi-metric summary (Accuracy, Loss, Latency, VRAM).
2. **Accuracy Comparison**: [../../experiments/plots/cifar_stem_accuracy_comparison.png](../../experiments/plots/cifar_stem_accuracy_comparison.png) - Test Accuracy comparison across Frozen & Finetuned modes.
3. **Loss Landscape**: [../../experiments/plots/cifar_stem_loss_comparison.png](../../experiments/plots/cifar_stem_loss_comparison.png) - Cross-Entropy Loss analysis and spatial feature misalignment effects.
4. **Throughput & Speedup**: [../../experiments/plots/cifar_stem_throughput_speedup.png](../../experiments/plots/cifar_stem_throughput_speedup.png) - Execution latency per epoch (s/epoch) and processing throughput (FPS).
5. **Peak VRAM Footprint**: [../../experiments/plots/cifar_stem_vram_memory_footprint.png](../../experiments/plots/cifar_stem_vram_memory_footprint.png) - Memory savings (~105.9 MB VRAM) without 224x224 upsampling.
6. **Parameter Breakdown**: [../../experiments/plots/cifar_stem_parameter_breakdown.png](../../experiments/plots/cifar_stem_parameter_breakdown.png) - Total vs Trainable parameter counts.
7. **Accuracy vs Latency Tradeoff**: [../../experiments/plots/cifar_stem_tradeoff_scatter.png](../../experiments/plots/cifar_stem_tradeoff_scatter.png) - Scatter plot evaluating classification accuracy against training latency.

---

## 💡 Key Takeaways & Recommendations

1. **Computational & Memory Gains:**
   - Native $32 \times 32$ stem achieves **~40% faster execution time per epoch** (48.93s vs 81.41s for finetuning).
   - Reduces Peak VRAM usage by **~105 MB (~15%)**.

2. **Accuracy & Convergence Tradeoff:**
   - `224x224` upsampling allows standard ImageNet pretrained weights to work out-of-the-box with high accuracy (**91.73%** fine-tuned after 3 epochs).
   - `32x32` modified stem alters `conv1` kernel ($3 \times 3$ vs $7 \times 7$) and removes maxpooling, breaking pretrained spatial feature alignment in frozen mode (48.28%). Finetuning recovers accuracy to **72.93%** in 3 epochs, but requires more epochs to match the upsampled baseline.

3. **Recommendation:**
   - Use `ResNet18_224_Finetune` when maximum accuracy is required.
   - Use `ResNet18_32_NativeStem_Finetune` for high-throughput edge deployment or resource-constrained rapid iteration after sufficient epoch fine-tuning.

---

## 💻 Implementation Snippet (PyTorch)

```python
import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet18_cifar_stem(mode="frozen", num_classes=10, device=None):
    """
    Build ResNet18 adapted with a 3x3 Conv1 stem for native 32x32 CIFAR-10 images.
    """
    weights = models.ResNet18_Weights.DEFAULT if mode in ["frozen", "finetune"] else None
    model = models.resnet18(weights=weights)
    
    # 1. Replace 7x7 stride=2 conv1 with 3x3 stride=1 conv1 for 32x32 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Bypass MaxPool to preserve spatial dims (32x32) into Layer 1
    model.maxpool = nn.Identity()
    
    # 3. Replace final classification layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 4. Apply freezing / fine-tuning policy
    if mode == "frozen":
        for name, param in model.named_parameters():
            if "fc" not in name and "conv1" not in name:
                param.requires_grad = False
    elif mode == "finetune":
        # Unfreeze conv1 + layer4 + fc
        for name, param in model.named_parameters():
            if not ("fc" in name or "layer4" in name or "conv1" in name):
                param.requires_grad = False
                
    if device is not None:
        model = model.to(device)
        
    return model
```

---

## 🎯 Success Criteria & Next Steps

1. **Success Condition:** Experiment executed and comparison matrix established.
2. **Next Steps:**
   - Stem builder helper implemented in `src/models/build_model.py`.
   - Results logged in [agents/experiments/CIFAR_STEM_EXPERIMENT.md](./CIFAR_STEM_EXPERIMENT.md) and [progress/MODEL_STATUS.md](../progress/MODEL_STATUS.md).

