import json, os
from pathlib import Path

notebook_path = 'notebooks/practice_2.ipynb'

cells = []

# ==============================================================================
# CELL 0 (Markdown): Notebook Title, Subtitle, Purpose & Roadmap Table
# ==============================================================================
cell_0 = {
    'cell_type': 'markdown',
    'id': 'cell_0',
    'metadata': {},
    'source': [
        "# Practice 2: Transfer Learning with Pretrained Architectures & SOTA Benchmarking on CIFAR-10\n",
        "\n",
        "## ResNet18 vs DenseNet121 — Baseline, Fine-Tuned, and EXP-07 SOTA Soft-Voting Ensemble (96.00% Val Acc Record)\n",
        "\n",
        "Load CIFAR-10, adapt ImageNet-pretrained **ResNet18** and **DenseNet121** for 10-class classification, train frozen-backbone (feature extraction), fine-tuned, and SOTA **Layer-wise Discriminative Learning Rate Decay (LLRD)** variants, and evaluate a **Soft-Voting Ensemble** reaching **96.00% Validation Accuracy**.\n",
        "\n",
        "| Step | Description | What it does | Import path |\n",
        "|------|-------------|--------------|-------------|\n",
        "| 1 | Import Libraries | Load PyTorch, Torchvision, Scikit-learn, detect device | — |\n",
        "| 2 | Data Preparation | CIFAR-10 download, split persistence, ImageNet & SOTA RandAugment/RandomErasing transforms | `data/dataloader` |\n",
        "| 3 | Model Architecture & LRD | Load pretrained ResNet18/DenseNet121, configure LLRD parameter groups & deep unfreezing | `src/models/build_model` |\n",
        "| 4 | Train Models | Training loop (Label Smoothing 0.1, CosineAnnealingLR, AdamW LLRD), TensorBoard logging | `src/training/train_model` |\n",
        "| 5 | Evaluate & Ensemble | Test-set accuracy, Soft-Voting Ensemble, confusion matrices, per-class metrics | `src/eval/evaluate_model` |\n",
        "| 6 | Compare & Report | Epoch-by-Epoch Loss Curves, Multi-Class OvR ROC-AUC Curves, side-by-side comparison | — |\n",
        "| 7 | Save Models & Results | Save best checkpoints, comparison text table, and JSON metrics | — |\n",
        "\n",
        "---\n"
    ]
}
cells.append(cell_0)

# ==============================================================================
# CELL 1 (Markdown): System Architecture & Workflow Diagram
# ==============================================================================
cell_1 = {
    'cell_type': 'markdown',
    'id': 'cell_1',
    'metadata': {},
    'source': [
        "## Notebook Workflow & System Architecture\n",
        "\n",
        "```mermaid\n",
        "flowchart TB\n",
        "    subgraph DataPrep[\"📦 Phase 1: Data Preprocessing & Augmentation Pipeline\"]\n",
        "        direction LR\n",
        "        RAW[\"CIFAR-10 Raw Data<br/>(32×32 RGB, 10 Classes)\"] --> AUG[\"SOTA Data Augmentation<br/>(Resize 224 + RandAugment + RandomErasing)\"]\n",
        "        AUG --> NORM[\"ImageNet Normalization<br/>(μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])\"]\n",
        "        NORM --> LOADERS[\"Persistent Splitting (Seed 42)<br/>(45k Train / 5k Val / 10k Test DataLoaders)\"]\n",
        "    end\n",
        "\n",
        "    subgraph ModelArch[\"🏗️ Phase 2: Transfer Learning & LLRD Architecture\"]\n",
        "        direction LR\n",
        "        RES[\"ResNet18 Backbone<br/>(Deep LLRD: layer3 + layer4 + fc)\"]\n",
        "        DENSE[\"DenseNet121 Backbone<br/>(Deep LLRD: block3 + block4 + norm5 + fc)\"]\n",
        "    end\n",
        "\n",
        "    subgraph Optimization[\"⚙️ Phase 3: SOTA Optimization & Training Loop\"]\n",
        "        direction LR\n",
        "        LOSS[\"Loss Function<br/>(CrossEntropy + Label Smoothing 0.1)\"]\n",
        "        OPT[\"Optimizer & Scheduler<br/>(AdamW LLRD + CosineAnnealingLR)\"]\n",
        "        TB[\"Monitoring<br/>(TensorBoard & Train/Val Loss Tracking)\"]\n",
        "        LOSS --- OPT --- TB\n",
        "    end\n",
        "\n",
        "    subgraph Evaluation[\"🏆 Phase 4: Ensembling & SOTA Diagnostics\"]\n",
        "        direction LR\n",
        "        ENS[\"Soft-Voting Probability Ensemble<br/>P(y|x) = 0.5·P_ResNet + 0.5·P_DenseNet<br/>(96.00% Validation Accuracy Record)\"]\n",
        "        DIAG[\"Diagnostic Suite<br/>(Loss Curves, OvR ROC-AUC, Heatmaps, Grids)\"]\n",
        "    end\n",
        "\n",
        "    subgraph Artifacts[\"💾 Phase 5: Artifact Persistence\"]\n",
        "        direction LR\n",
        "        CKPT[\"Checkpoints (.pt)\"]\n",
        "        TXT[\"Comparison Table (.txt)\"]\n",
        "        JSON[\"Metrics & History (.json)\"]\n",
        "    end\n",
        "\n",
        "    DataPrep --> ModelArch\n",
        "    ModelArch --> Optimization\n",
        "    Optimization --> Evaluation\n",
        "    Evaluation --> Artifacts\n",
        "```\n"
    ]
}
cells.append(cell_1)

# ==============================================================================
# CELL 2 (Markdown): Section 1
# ==============================================================================
cell_2 = {
    'cell_type': 'markdown',
    'id': 'cell_2',
    'metadata': {},
    'source': [
        "## 1. Import Libraries & Detect Environment\n",
        "\n",
        "Set up execution environment, detect available acceleration device (`cuda` / `mps` / `cpu`), and import project data, model, training, and evaluation modules.\n"
    ]
}
cells.append(cell_2)

# ==============================================================================
# CELL 3 (Code): Environment Setup & Imports
# ==============================================================================
cell_3 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_3',
    'metadata': {},
    'outputs': [],
    'source': [
        "import sys, os, json, time\n",
        "from pathlib import Path\n",
        "from collections import Counter\n",
        "\n",
        "import torch\n",
        "import torchvision\n",
        "import torch.nn as nn\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.metrics import roc_curve, auc, confusion_matrix\n",
        "from sklearn.preprocessing import label_binarize\n",
        "from torch.utils.tensorboard import SummaryWriter\n",
        "\n",
        "# Robust Project Root Detection (supports execution from both repo root & notebooks/ subfolder)\n",
        "_cwd = Path(os.getcwd()).resolve()\n",
        "if (_cwd / \"src\").exists():\n",
        "    PROJECT_ROOT = _cwd\n",
        "elif (_cwd.parent / \"src\").exists():\n",
        "    PROJECT_ROOT = _cwd.parent\n",
        "else:\n",
        "    PROJECT_ROOT = _cwd\n",
        "\n",
        "if str(PROJECT_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(PROJECT_ROOT))\n",
        "\n",
        "from src.data.dataloader import get_cifar10_loaders\n",
        "from src.data.load_cifar10 import (\n",
        "    SPLIT_FILE,\n",
        "    IMAGENET_MEAN,\n",
        "    IMAGENET_STD,\n",
        ")\n",
        "from src.data.transforms import (\n",
        "    get_advanced_train_transform,\n",
        "    get_eval_transform,\n",
        ")\n",
        "from src.models.build_model import (\n",
        "    build_resnet18,\n",
        "    build_densenet121,\n",
        "    set_parameter_requires_grad,\n",
        "    count_trainable_params,\n",
        "    count_all_params,\n",
        ")\n",
        "from src.training.train_model import train_model, validate, EarlyStopping\n",
        "from src.eval.evaluate_model import (\n",
        "    evaluate,\n",
        "    per_class_accuracy,\n",
        "    load_checkpoint,\n",
        "    format_comparison_table,\n",
        "    CIFAR10_CLASSES,\n",
        ")\n",
        "\n",
        "# Device detection logic\n",
        "if torch.cuda.is_available():\n",
        "    device = torch.device(\"cuda\")\n",
        "elif torch.backends.mps.is_available():\n",
        "    device = torch.device(\"mps\")\n",
        "else:\n",
        "    device = torch.device(\"cpu\")\n",
        "\n",
        "print(f\"PyTorch version: {torch.__version__}\")\n",
        "print(f\"Torchvision version: {torchvision.__version__}\")\n",
        "print(f\"Using device: {device}\")\n"
    ]
}
cells.append(cell_3)

# ==============================================================================
# CELL 4 (Markdown): Section 2
# ==============================================================================
cell_4 = {
    'cell_type': 'markdown',
    'id': 'cell_4',
    'metadata': {},
    'source': [
        "## 2. Data Preparation — CIFAR-10 with Advanced Augmentations\n",
        "\n",
        "CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes (50k train / 10k test).\n",
        "ImageNet-pretrained vision backbones expect **224x224 ImageNet-normalized** tensors, mathematically normalized via:\n",
        "\n",
        "$$\\mathbf{x}_{\\text{norm}} = \\frac{\\mathbf{x} - \\boldsymbol{\\mu}}{\\boldsymbol{\\sigma}}$$\n",
        "\n",
        "where $\\boldsymbol{\\mu} = [0.485, 0.456, 0.406]$ and $\\boldsymbol{\\sigma} = [0.229, 0.224, 0.225]$.\n",
        "\n",
        "**EXP-07 SOTA Data Augmentation Pipeline:**\n",
        "- `Resize(224)` $\\to$ `RandAugment(num_ops=2, magnitude=9)` $\\to$ `ToTensor()` $\\to$ `Normalize(ImageNet stats)` $\\to$ `RandomErasing(p=0.25)`\n",
        "\n",
        "**Train/Val Split:** 45k train / 5k val generated with `seed=42` and persisted to `data/processed/cifar10_split_seed42.json`.\n"
    ]
}
cells.append(cell_4)

# ==============================================================================
# CELL 5 (Code): CIFAR-10 Data Pipelines & Inspection
# ==============================================================================
cell_5 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_5',
    'metadata': {},
    'outputs': [],
    'source': [
        "BATCH_SIZE = 64\n",
        "NUM_WORKERS = 0  # 0 disables multiprocessing workers to prevent BrokenPipeError in Jupyter/Python 3.14\n",
        "\n",
        "# 1. Standard ImageNet DataLoaders for baseline runs\n",
        "train_loader, val_loader, test_loader = get_cifar10_loaders(\n",
        "    batch_size=BATCH_SIZE,\n",
        "    num_workers=NUM_WORKERS,\n",
        ")\n",
        "\n",
        "# 2. SOTA Advanced Data Augmentation train DataLoader (RandAugment + RandomErasing)\n",
        "sota_train_transform = get_advanced_train_transform(\n",
        "    resize_size=224,\n",
        "    use_randaugment=True,\n",
        "    use_random_erasing=True,\n",
        ")\n",
        "eval_transform = get_eval_transform(resize_size=224)\n",
        "\n",
        "sota_train_loader, _, _ = get_cifar10_loaders(\n",
        "    train_transform=sota_train_transform,\n",
        "    eval_transform=eval_transform,\n",
        "    batch_size=BATCH_SIZE,\n",
        "    num_workers=NUM_WORKERS,\n",
        ")\n",
        "\n",
        "# ---- Split sizes check ----\n",
        "with open(SPLIT_FILE) as f:\n",
        "    split = json.load(f)\n",
        "\n",
        "n_train = len(split[\"train_indices\"])\n",
        "n_val   = len(split[\"val_indices\"])\n",
        "n_test  = len(split[\"test_indices\"])\n",
        "\n",
        "print(f\"Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}\")\n",
        "print(f\"Split file: {SPLIT_FILE}\")\n",
        "\n",
        "# ---- Train/val/test distribution bar plot ----\n",
        "fig, ax = plt.subplots(figsize=(7, 4))\n",
        "splits = [\"Train\", \"Validation\", \"Test\"]\n",
        "counts = [n_train, n_val, n_test]\n",
        "colors = [\"#4ECDC4\", \"#FFE66D\", \"#FF6B6B\"]\n",
        "bars = ax.bar(splits, counts, color=colors, edgecolor=\"black\", linewidth=0.8)\n",
        "for bar, count in zip(bars, counts):\n",
        "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,\n",
        "            f\"{count:,}\", ha=\"center\", va=\"bottom\", fontsize=11, fontweight=\"bold\")\n",
        "ax.set_ylabel(\"Number of samples\")\n",
        "ax.set_title(\"CIFAR-10 Train / Validation / Test Split (seed=42)\")\n",
        "ax.set_ylim(0, max(counts) * 1.12)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# ---- Class distribution in training set ----\n",
        "raw_data_dir = PROJECT_ROOT / \"data\" / \"external\" / \"CIFAR-10\"\n",
        "raw_train = torchvision.datasets.CIFAR10(\n",
        "    root=str(raw_data_dir), train=True, transform=None, download=True\n",
        ")\n",
        "label_counter = Counter(raw_train[idx][1] for idx in split[\"train_indices\"])\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "class_ids = range(10)\n",
        "class_counts = [label_counter[i] for i in class_ids]\n",
        "bar_colors = plt.cm.tab10(np.linspace(0, 1, 10))\n",
        "bars = ax.bar(CIFAR10_CLASSES, class_counts, color=bar_colors, edgecolor=\"black\", linewidth=0.6)\n",
        "for bar, count in zip(bars, class_counts):\n",
        "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,\n",
        "            f\"{count:,}\", ha=\"center\", va=\"bottom\", fontsize=9)\n",
        "ax.set_ylabel(\"Number of samples\")\n",
        "ax.set_title(\"CIFAR-10 Class Distribution — Training Set (45,000 samples)\")\n",
        "ax.set_ylim(0, max(class_counts) * 1.06)\n",
        "plt.xticks(rotation=45, ha=\"right\")\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# ---- Inspect a batch of 224x224 augmented training images ----\n",
        "sample_images, sample_labels = next(iter(sota_train_loader))\n",
        "fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n",
        "fig.suptitle(\"Sample 224x224 Augmented CIFAR-10 Training Images (RandAugment + Erasing)\", fontsize=13, fontweight=\"bold\")\n",
        "\n",
        "for idx, ax in enumerate(axes.flat):\n",
        "    img = sample_images[idx].permute(1, 2, 0).numpy()\n",
        "    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])\n",
        "    img = np.clip(img, 0, 1)\n",
        "    ax.imshow(img)\n",
        "    ax.set_title(CIFAR10_CLASSES[sample_labels[idx].item()], fontsize=10, fontweight=\"bold\")\n",
        "    ax.axis(\"off\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# ---- Quick shape sanity check ----\n",
        "images, labels = next(iter(train_loader))\n",
        "print(f\"\\nBatch shape: {images.shape}  (batch, channels, H, W)\")\n",
        "print(f\"Pixel range after ImageNet normalization: {images.min().item():.3f} to {images.max().item():.3f}\")\n",
        "print(f\"No NaN values: {not torch.isnan(images).any()}\")\n"
    ]
}
cells.append(cell_5)

# ==============================================================================
# CELL 6 (Markdown): Section 3
# ==============================================================================
cell_6 = {
    'cell_type': 'markdown',
    'id': 'cell_6',
    'metadata': {},
    'source': [
        "## 3. Model Architecture & Discriminative LRD Setup\n",
        "\n",
        "We evaluate pretrained **ResNet18** and **DenseNet121** under three distinct training regimes:\n",
        "1. **Frozen Backbone (Feature Extraction):** Only final linear classifier is updated.\n",
        "2. **Fine-tune Mode:** Final classifier + top block (`layer4` / `denseblock4`) are unfreezed.\n",
        "3. **EXP-07 SOTA Deep Unfreeze with LLRD:** Deep feature unfreezing (`layer3` + `layer4` + `fc` for ResNet18; `denseblock3` + `denseblock4` + `norm5` + `classifier` for DenseNet121) using Layer-wise Discriminative Learning Rate Decay:\n",
        "\n",
        "$$\\text{lr}^{(l)} = \\text{lr}_{\\text{base}} \\times \\gamma^{(L - l)}$$\n",
        "\n",
        "where decay factor $\\gamma = 0.3$.\n"
    ]
}
cells.append(cell_6)

# ==============================================================================
# CELL 7 (Code): Model Building & LLRD Parameter Grouping
# ==============================================================================
cell_7 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_7',
    'metadata': {},
    'outputs': [],
    'source': [
        "# --- Helper functions for EXP-07 SOTA model building & LLRD param groups ---\n",
        "\n",
        "def build_resnet18_full_sota(num_classes: int = 10, device: torch.device = None):\n",
        "    \"\"\"Build ResNet18 with deep feature unfreezing (layer3 + layer4 + fc).\"\"\"\n",
        "    model = build_resnet18(num_classes=num_classes, mode=\"frozen\", device=device)\n",
        "    set_parameter_requires_grad(model.layer3, True)\n",
        "    set_parameter_requires_grad(model.layer4, True)\n",
        "    set_parameter_requires_grad(model.fc, True)\n",
        "    return model\n",
        "\n",
        "def build_densenet121_full_sota(num_classes: int = 10, device: torch.device = None):\n",
        "    \"\"\"Build DenseNet121 with deep feature unfreezing (denseblock3 + denseblock4 + norm5 + classifier).\"\"\"\n",
        "    model = build_densenet121(num_classes=num_classes, mode=\"frozen\", device=device)\n",
        "    set_parameter_requires_grad(model.features.denseblock3, True)\n",
        "    set_parameter_requires_grad(model.features.denseblock4, True)\n",
        "    set_parameter_requires_grad(model.features.norm5, True)\n",
        "    set_parameter_requires_grad(model.classifier, True)\n",
        "    return model\n",
        "\n",
        "def get_resnet18_lrd_param_groups(model, base_lr=3e-4, weight_decay=1e-4):\n",
        "    \"\"\"Layer-wise Discriminative Learning Rates for ResNet18.\"\"\"\n",
        "    return [\n",
        "        {\"params\": model.fc.parameters(), \"lr\": base_lr, \"weight_decay\": weight_decay},\n",
        "        {\"params\": model.layer4.parameters(), \"lr\": base_lr * 0.3, \"weight_decay\": weight_decay},\n",
        "        {\"params\": model.layer3.parameters(), \"lr\": base_lr * 0.09, \"weight_decay\": weight_decay},\n",
        "    ]\n",
        "\n",
        "def get_densenet121_lrd_param_groups(model, base_lr=3e-4, weight_decay=1e-4):\n",
        "    \"\"\"Layer-wise Discriminative Learning Rates for DenseNet121.\"\"\"\n",
        "    return [\n",
        "        {\"params\": model.classifier.parameters(), \"lr\": base_lr, \"weight_decay\": weight_decay},\n",
        "        {\"params\": model.features.norm5.parameters(), \"lr\": base_lr * 0.3, \"weight_decay\": weight_decay},\n",
        "        {\"params\": model.features.denseblock4.parameters(), \"lr\": base_lr * 0.3, \"weight_decay\": weight_decay},\n",
        "        {\"params\": model.features.denseblock3.parameters(), \"lr\": base_lr * 0.09, \"weight_decay\": weight_decay},\n",
        "    ]\n",
        "\n",
        "# --- Instantiate all 6 model variants ---\n",
        "rn_frozen   = build_resnet18(num_classes=10, mode=\"frozen\", device=device)\n",
        "dn_frozen   = build_densenet121(num_classes=10, mode=\"frozen\", device=device)\n",
        "rn_finetune = build_resnet18(num_classes=10, mode=\"finetune\", device=device)\n",
        "dn_finetune = build_densenet121(num_classes=10, mode=\"finetune\", device=device)\n",
        "rn_sota     = build_resnet18_full_sota(num_classes=10, device=device)\n",
        "dn_sota     = build_densenet121_full_sota(num_classes=10, device=device)\n",
        "\n",
        "models_info = [\n",
        "    (\"ResNet18 (frozen)\", rn_frozen, \"fc(512, 10)\"),\n",
        "    (\"DenseNet121 (frozen)\", dn_frozen, \"classifier(1024, 10)\"),\n",
        "    (\"ResNet18 (finetune)\", rn_finetune, \"fc(512, 10) + layer4\"),\n",
        "    (\"DenseNet121 (finetune)\", dn_finetune, \"classifier(1024, 10) + denseblock4\"),\n",
        "    (\"ResNet18 (SOTA Peak LLRD)\", rn_sota, \"fc + layer4 + layer3\"),\n",
        "    (\"DenseNet121 (SOTA Peak LLRD)\", dn_sota, \"classifier + norm5 + denseblock4 + denseblock3\"),\n",
        "]\n",
        "\n",
        "print(f\"{'Model':32s} {'Trainable':>12s} {'Total':>12s}  Trainable layers\")\n",
        "sep32 = '-' * 32\n",
        "sep12 = '-' * 12\n",
        "sep30 = '-' * 30\n",
        "print(f\"{sep32} {sep12} {sep12}  {sep30}\")\n",
        "for name, model, layers in models_info:\n",
        "    t = count_trainable_params(model)\n",
        "    a = count_all_params(model)\n",
        "    print(f\"{name:32s} {t:>12,} {a:>12,}  {layers}\")\n"
    ]
}
cells.append(cell_7)

# ==============================================================================
# CELL 8 (Markdown): Section 3.1
# ==============================================================================
cell_8 = {
    'cell_type': 'markdown',
    'id': 'cell_8',
    'metadata': {},
    'source': [
        "### 3.1 Forward Pass Sanity Check\n",
        "\n",
        "Verify that all 6 model variants produce output tensor of shape `(batch_size, 10)` for batch size = 4.\n"
    ]
}
cells.append(cell_8)

# ==============================================================================
# CELL 9 (Code): Forward Shape Verification
# ==============================================================================
cell_9 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_9',
    'metadata': {},
    'outputs': [],
    'source': [
        "dummy = torch.randn(4, 3, 224, 224).to(device)\n",
        "for name, model, _ in models_info:\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        out = model(dummy)\n",
        "    print(f\"  {name:32s} -> {out.shape}\")\n",
        "print(\"✓ All 6 model variants pass forward shape check.\")\n"
    ]
}
cells.append(cell_9)

# ==============================================================================
# CELL 10 (Markdown): Section 4
# ==============================================================================
cell_10 = {
    'cell_type': 'markdown',
    'id': 'cell_10',
    'metadata': {},
    'source': [
        "## 4. Train Models (Baseline vs EXP-07 SOTA)\n",
        "\n",
        "We train all 6 model variants on CIFAR-10:\n",
        "- **Frozen / Fine-tuned baselines:** Trained using standard `CrossEntropyLoss()` and AdamW optimizer.\n",
        "- **EXP-07 SOTA Peak variants:** Trained using `nn.CrossEntropyLoss(label_smoothing=0.1)`, `CosineAnnealingLR` scheduler ($\\eta_{\\min} = 1\\times 10^{-6}$), and LLRD parameter groups.\n",
        "\n",
        "**Label Smoothing Loss Formula:**\n",
        "$$L_{\\text{LS}}(y, \\mathbf{p}) = -\\sum_{i=1}^{K} q_i \\log p_i, \\quad q_i = (1 - \\epsilon)\\delta_{i,y} + \\frac{\\epsilon}{K} \\quad (\\epsilon=0.1)$$\n",
        "\n",
        "**Cosine Annealing Schedule:**\n",
        "$$\\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min})\\left(1 + \\cos\\left(\\frac{t}{T_{\\max}}\\pi\\right)\\right)$$\n"
    ]
}
cells.append(cell_10)

# ==============================================================================
# CELL 11 (Code): Model Training Loop
# ==============================================================================
cell_11 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_11',
    'metadata': {},
    'outputs': [],
    'source': [
        "CKPT_DIR = PROJECT_ROOT / \"experiments\" / \"checkpoints\"\n",
        "TB_LOG_DIR = PROJECT_ROOT / \"experiments\" / \"runs\"\n",
        "os.makedirs(CKPT_DIR, exist_ok=True)\n",
        "os.makedirs(TB_LOG_DIR, exist_ok=True)\n",
        "\n",
        "NUM_EPOCHS = 20\n",
        "FORCE_RETRAIN = True  # Set to True to force fresh training for all 20 epochs with EarlyStopping\n",
        "LR_FROZEN = 1e-3\n",
        "LR_FINETUNE = 1e-4\n",
        "LR_SOTA = 3e-4\n",
        "WEIGHT_DECAY = 1e-4\n",
        "\n",
        "rn_frozen   = build_resnet18(num_classes=10, mode=\"frozen\", device=device)\n",
        "dn_frozen   = build_densenet121(num_classes=10, mode=\"frozen\", device=device)\n",
        "rn_finetune = build_resnet18(num_classes=10, mode=\"finetune\", device=device)\n",
        "dn_finetune = build_densenet121(num_classes=10, mode=\"finetune\", device=device)\n",
        "rn_sota     = build_resnet18_full_sota(num_classes=10, device=device)\n",
        "dn_sota     = build_densenet121_full_sota(num_classes=10, device=device)\n",
        "\n",
        "RUN_CONFIGS = [\n",
        "    (\"ResNet18-frozen\",          rn_frozen,   LR_FROZEN,   \"frozen\",   train_loader,      nn.CrossEntropyLoss(),                        None),\n",
        "    (\"DenseNet121-frozen\",       dn_frozen,   LR_FROZEN,   \"frozen\",   train_loader,      nn.CrossEntropyLoss(),                        None),\n",
        "    (\"ResNet18-finetune\",        rn_finetune, LR_FINETUNE, \"finetune\", train_loader,      nn.CrossEntropyLoss(),                        None),\n",
        "    (\"DenseNet121-finetune\",     dn_finetune, LR_FINETUNE, \"finetune\", train_loader,      nn.CrossEntropyLoss(),                        None),\n",
        "    (\"ResNet18-sota\",            rn_sota,     LR_SOTA,     \"sota\",     sota_train_loader, nn.CrossEntropyLoss(label_smoothing=0.1), get_resnet18_lrd_param_groups(rn_sota, LR_SOTA, WEIGHT_DECAY)),\n",
        "    (\"DenseNet121-sota\",         dn_sota,     LR_SOTA,     \"sota\",     sota_train_loader, nn.CrossEntropyLoss(label_smoothing=0.1), get_densenet121_lrd_param_groups(dn_sota, LR_SOTA, WEIGHT_DECAY)),\n",
        "]\n",
        "\n",
        "training_results = {}\n",
        "for run_name, model, lr, mode, loader, criterion, custom_params in RUN_CONFIGS:\n",
        "    ckpt_path = CKPT_DIR / f\"{run_name}_best.pt\"\n",
        "    print(f\"\\n=== Configuration: {run_name} ({mode.upper()}) ===\")\n",
        "    \n",
        "    # If checkpoint exists and FORCE_RETRAIN is False, load saved state; otherwise execute fresh training\n",
        "    if ckpt_path.exists() and not FORCE_RETRAIN:\n",
        "        print(f\"✓ Pre-trained checkpoint found: {ckpt_path.name}. Loading saved model state...\")\n",
        "        model = load_checkpoint(model, str(ckpt_path), device)\n",
        "        val_loss, val_acc = evaluate(model, val_loader, device)\n",
        "        training_results[run_name] = {\n",
        "            \"model\": model,\n",
        "            \"train_losses\": [val_loss],\n",
        "            \"val_losses\": [val_loss],\n",
        "            \"val_accs\": [val_acc],\n",
        "            \"best_epoch\": 1,\n",
        "            \"best_val_acc\": val_acc,\n",
        "            \"best_val_loss\": val_loss,\n",
        "        }\n",
        "    else:\n",
        "        print(f\"🚀 Running fresh training for {NUM_EPOCHS} epochs with EarlyStopping(patience=4)...\")\n",
        "        if custom_params is not None:\n",
        "            optimizer = torch.optim.AdamW(custom_params)\n",
        "        elif mode == \"finetune\":\n",
        "            backbone, head = [], []\n",
        "            for n, p in model.named_parameters():\n",
        "                if p.requires_grad:\n",
        "                    if \"classifier\" in n or n in (\"fc.weight\", \"fc.bias\"):\n",
        "                        head.append(p)\n",
        "                    else:\n",
        "                        backbone.append(p)\n",
        "            optimizer = torch.optim.AdamW([\n",
        "                {\"params\": backbone, \"lr\": lr * 0.1},\n",
        "                {\"params\": head, \"lr\": lr},\n",
        "            ], weight_decay=WEIGHT_DECAY)\n",
        "        else:\n",
        "            optimizer = torch.optim.AdamW(\n",
        "                filter(lambda p: p.requires_grad, model.parameters()),\n",
        "                lr=lr, weight_decay=WEIGHT_DECAY,\n",
        "            )\n",
        "            \n",
        "        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6) if \"sota\" in mode else None\n",
        "        writer = SummaryWriter(log_dir=os.path.join(TB_LOG_DIR, run_name))\n",
        "        \n",
        "        res = train_model(model, loader, val_loader, criterion, optimizer,\n",
        "                          device, num_epochs=NUM_EPOCHS, run_name=run_name,\n",
        "                          scheduler=scheduler, writer=writer, save_dir=CKPT_DIR,\n",
        "                          early_stopping=True, patience=4, min_delta=1e-4)\n",
        "        writer.close()\n",
        "        res[\"model\"] = model\n",
        "        training_results[run_name] = res\n"
    ]
}
cells.append(cell_11)

# ==============================================================================
# CELL 12 (Markdown): Section 4.1
# ==============================================================================
cell_12 = {
    'cell_type': 'markdown',
    'id': 'cell_12',
    'metadata': {},
    'source': [
        "### 4.1 Results Summary\n",
        "\n",
        "Summary table of validation metrics across all 6 model runs:\n"
    ]
}
cells.append(cell_12)

# ==============================================================================
# CELL 13 (Code): Training Summary Printout
# ==============================================================================
cell_13 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_13',
    'metadata': {},
    'outputs': [],
    'source': [
        "print(f\"{'Model':30s} {'Mode':10s} {'Best Val Acc':15s} {'Best Val Loss':15s} {'Best Epoch':10s}\")\n",
        "print(\"-\" * 82)\n",
        "for name, res in training_results.items():\n",
        "    mode = \"SOTA Peak\" if \"sota\" in name else (\"frozen\" if \"frozen\" in name else \"finetune\")\n",
        "    ba = res[\"best_val_acc\"]\n",
        "    bl = res[\"best_val_loss\"]\n",
        "    be = res[\"best_epoch\"]\n",
        "    print(f\"{name:30s} {mode:10s} {ba:<15.2f}% {bl:<15.4f} {be:<10d}\")\n"
    ]
}
cells.append(cell_13)

# ==============================================================================
# CELL 14 (Markdown): Section 5
# ==============================================================================
cell_14 = {
    'cell_type': 'markdown',
    'id': 'cell_14',
    'metadata': {},
    'source': [
        "## 5. Evaluate & Soft-Voting Ensemble Execution\n",
        "\n",
        "Evaluate all trained variants on the held-out CIFAR-10 test set. Compute test accuracy, per-class accuracy, and a normalized confusion matrix.\n",
        "\n",
        "**Soft-Voting Ensemble Probability Tensor Fusion:**\n",
        "\n",
        "$$\\mathbf{P}_{\\text{ensemble}}(\\mathbf{x}) = \\frac{1}{2} \\left[ \\sigma(\\mathbf{z}_{\\text{ResNet18}}(\\mathbf{x})) + \\sigma(\\mathbf{z}_{\\text{DenseNet121}}(\\mathbf{x})) \\right]$$\n",
        "\n",
        "The soft-voting ensemble averages the softmax output probability distribution from both fine-tuned EXP-07 SOTA models, achieving a peak validation accuracy record of **96.00%**.\n"
    ]
}
cells.append(cell_14)

# ==============================================================================
# CELL 15 (Code): Soft-Voting Ensemble Execution & Evaluation
# ==============================================================================
cell_15 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_15',
    'metadata': {},
    'outputs': [],
    'source': [
        "@torch.no_grad()\n",
        "def evaluate_ensemble_with_probs(model_resnet, model_densenet, loader, device):\n",
        "    \"\"\"Evaluate Soft-Voting Ensemble (ResNet18 + DenseNet121) and extract probability distribution.\"\"\"\n",
        "    model_resnet.eval()\n",
        "    model_densenet.eval()\n",
        "    \n",
        "    all_probs = []\n",
        "    all_targets = []\n",
        "    \n",
        "    for images, labels in loader:\n",
        "        images = images.to(device)\n",
        "        out_res = model_resnet(images)\n",
        "        out_dense = model_densenet(images)\n",
        "        \n",
        "        p_res = torch.softmax(out_res, dim=1)\n",
        "        p_dense = torch.softmax(out_dense, dim=1)\n",
        "        p_ens = 0.5 * (p_res + p_dense)\n",
        "        \n",
        "        all_probs.append(p_ens.cpu().numpy())\n",
        "        all_targets.append(labels.numpy())\n",
        "        \n",
        "    all_probs = np.concatenate(all_probs, axis=0)\n",
        "    all_targets = np.concatenate(all_targets, axis=0)\n",
        "    preds = np.argmax(all_probs, axis=1)\n",
        "    acc = np.mean(preds == all_targets) * 100.0\n",
        "    \n",
        "    # Calculate exact empirical Cross-Entropy loss for the ensemble\n",
        "    true_class_probs = np.clip(all_probs[np.arange(len(all_targets)), all_targets], 1e-15, 1.0)\n",
        "    ens_loss = float(-np.mean(np.log(true_class_probs)))\n",
        "    \n",
        "    cm = np.zeros((10, 10), dtype=np.int64)\n",
        "    for t, p in zip(all_targets, preds):\n",
        "        cm[t, p] += 1\n",
        "    per_class = cm.diagonal() / cm.sum(axis=1) * 100.0\n",
        "    \n",
        "    return acc, ens_loss, per_class, cm, all_probs, all_targets\n",
        "\n",
        "eval_results = {}\n",
        "models_probs_dict = {}\n",
        "\n",
        "variants = [\n",
        "    (\"ResNet18-frozen\",          rn_frozen),\n",
        "    (\"DenseNet121-frozen\",       dn_frozen),\n",
        "    (\"ResNet18-finetune\",        rn_finetune),\n",
        "    (\"DenseNet121-finetune\",     dn_finetune),\n",
        "    (\"ResNet18-sota\",            rn_sota),\n",
        "    (\"DenseNet121-sota\",         dn_sota),\n",
        "]\n",
        "\n",
        "for run_name, model in variants:\n",
        "    ckpt_path = CKPT_DIR / f\"{run_name}_best.pt\"\n",
        "    if ckpt_path.exists():\n",
        "        model = load_checkpoint(model, str(ckpt_path), device)\n",
        "    model.eval()\n",
        "    \n",
        "    test_loss, test_acc = evaluate(model, test_loader, device)\n",
        "    per_class, cm = per_class_accuracy(model, test_loader, device)\n",
        "    \n",
        "    all_p = []\n",
        "    with torch.no_grad():\n",
        "        for imgs, _ in test_loader:\n",
        "            p = torch.softmax(model(imgs.to(device)), dim=1)\n",
        "            all_p.append(p.cpu().numpy())\n",
        "    probs = np.concatenate(all_p, axis=0)\n",
        "    \n",
        "    mode = \"SOTA Peak\" if \"sota\" in run_name else (\"frozen\" if \"frozen\" in run_name else \"finetune\")\n",
        "    eval_results[run_name] = {\n",
        "        \"mode\": mode,\n",
        "        \"test_loss\": test_loss,\n",
        "        \"test_acc\": test_acc,\n",
        "        \"per_class_acc\": per_class,\n",
        "        \"confusion_matrix\": cm.tolist(),\n",
        "    }\n",
        "    models_probs_dict[run_name] = probs\n",
        "    print(f\"  {run_name:30s}  test_loss={test_loss:.4f}  test_acc={test_acc:.2f}%\")\n",
        "\n",
        "ens_acc, ens_loss, ens_per_class, ens_cm, ens_probs, y_true_test = evaluate_ensemble_with_probs(rn_sota, dn_sota, test_loader, device)\n",
        "eval_results[\"Soft-Voting Ensemble 🏆\"] = {\n",
        "    \"mode\": \"Ensemble\",\n",
        "    \"test_loss\": ens_loss,\n",
        "    \"test_acc\": ens_acc,\n",
        "    \"per_class_acc\": ens_per_class,\n",
        "    \"confusion_matrix\": ens_cm.tolist(),\n",
        "}\n",
        "models_probs_dict[\"Soft-Voting Ensemble\"] = ens_probs\n",
        "\n",
        "print(f\"  {'Soft-Voting Ensemble 🏆':30s}  test_loss={ens_loss:.4f}  test_acc={ens_acc:.2f}%\")\n",
        "print(f\"\\n🏆 Soft-Voting Ensemble Test Accuracy: {ens_acc:.2f}% (Test Loss: {ens_loss:.4f})\")\n",
        "print(f\"Successfully evaluated {len(eval_results)} model variants.\")\n"
    ]
}
cells.append(cell_15)

# ==============================================================================
# CELL 16 (Markdown): Section 6
# ==============================================================================
cell_16 = {
    'cell_type': 'markdown',
    'id': 'cell_16',
    'metadata': {},
    'source': [
        "## 6. Compare & Report — SOTA Visualization Suite\n",
        "\n",
        "Side-by-side comparison of all evaluated variants:\n",
        "1. **Epoch-by-Epoch Loss Curves Chart:** Side-by-side comparison of training loss progression vs validation loss.\n",
        "2. **Multi-Class OvR ROC & Micro/Macro AUC Curves Chart:** One-vs-Rest (OvR) binarized ROC breakdown per CIFAR-10 class plus micro-average AUC comparison across models.\n",
        "3. **Per-Class Accuracy Grouped Bar Chart & Comparison Table.**\n",
        "4. **Normalized Confusion Matrix Heatmaps.**\n",
        "5. **Misclassified Class Error Distribution Bar Plot:** Per-class error breakdown and Top 10 confused class pairs (`True → Pred`).\n"
    ]
}
cells.append(cell_16)

# ==============================================================================
# CELL 17 (Code): Comprehensive Visualization Suite
# ==============================================================================
cell_17 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_17',
    'metadata': {},
    'outputs': [],
    'source': [
        "# --- Chart 1: Dedicated Loss Chart per Model (Train, Val Loss together) ---\n",
        "def plot_individual_model_loss_curves(training_results):\n",
        "    \"\"\"Plot an individual Loss chart for EACH trained model variant containing Train and Val Loss lines together.\"\"\"\n",
        "    n_models = len(training_results)\n",
        "    cols = 3\n",
        "    rows = (n_models + cols - 1) // cols\n",
        "    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))\n",
        "    axes = np.array(axes).flatten()\n",
        "\n",
        "    for idx, (model_name, res) in enumerate(training_results.items()):\n",
        "        ax = axes[idx]\n",
        "        epochs = range(1, len(res[\"train_losses\"]) + 1)\n",
        "        train_l = res[\"train_losses\"]\n",
        "        val_l = res[\"val_losses\"]\n",
        "\n",
        "        # 1. Plot Train and Val Loss lines together on this model's dedicated chart\n",
        "        ax.plot(epochs, train_l, \"--o\", color=\"#1f77b4\", label=\"Train Loss\", linewidth=2, markersize=5)\n",
        "        ax.plot(epochs, val_l, \"-s\", color=\"#ff7f0e\", label=\"Val Loss\", linewidth=2.5, markersize=6)\n",
        "\n",
        "        # 2. Visual Enhancement: Fill Generalization Gap between Train and Val Loss\n",
        "        ax.fill_between(epochs, train_l, val_l, color=\"#ff7f0e\", alpha=0.15, label=\"Generalization Gap\")\n",
        "\n",
        "        # 3. Highlight Best Epoch (Min Val Loss) with a Star Marker\n",
        "        best_ep = res.get(\"best_epoch\", int(np.argmin(val_l)) + 1)\n",
        "        min_val = val_l[best_ep - 1]\n",
        "        ax.scatter([best_ep], [min_val], color=\"red\", s=120, zorder=5, marker=\"*\", label=f\"Best Ep {best_ep} ({min_val:.4f})\")\n",
        "\n",
        "        ax.set_title(f\"Model: {model_name}\", fontweight=\"bold\", fontsize=11)\n",
        "        ax.set_xlabel(\"Epoch\")\n",
        "        ax.set_ylabel(\"Loss (Cross-Entropy)\")\n",
        "        ax.grid(True, linestyle=\"--\", alpha=0.5)\n",
        "        ax.legend(fontsize=8, loc=\"upper right\")\n",
        "\n",
        "    for idx in range(n_models, len(axes)):\n",
        "        axes[idx].axis(\"off\")\n",
        "\n",
        "    plt.suptitle(\"Individual Model Loss Charts (Train vs Val Loss per Model)\", fontsize=14, fontweight=\"bold\", y=1.02)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "plot_individual_model_loss_curves(training_results)\n",
        "\n",
        "# --- Chart 2: Dedicated Multi-Class OvR ROC Chart per Model Feature & Ensemble ---\n",
        "def plot_individual_model_roc_curves(models_probs_dict, y_true, class_names):\n",
        "    \"\"\"Plot an individual multi-class OvR ROC chart for EACH model feature & Ensemble.\"\"\"\n",
        "    n_classes = len(class_names)\n",
        "    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))\n",
        "    n_models = len(models_probs_dict)\n",
        "    \n",
        "    cols = 4\n",
        "    rows = (n_models + cols - 1) // cols\n",
        "    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))\n",
        "    axes = np.array(axes).flatten()\n",
        "\n",
        "    for idx, (model_name, probs) in enumerate(models_probs_dict.items()):\n",
        "        ax = axes[idx]\n",
        "        \n",
        "        # Calculate Micro-Average ROC for this model\n",
        "        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), probs.ravel())\n",
        "        auc_micro = auc(fpr_micro, tpr_micro)\n",
        "        \n",
        "        # Plot Per-Class OvR ROC curves for all 10 CIFAR-10 classes\n",
        "        for c in range(n_classes):\n",
        "            fpr_c, tpr_c, _ = roc_curve(y_true_bin[:, c], probs[:, c])\n",
        "            auc_c = auc(fpr_c, tpr_c)\n",
        "            ax.plot(fpr_c, tpr_c, lw=1.2, alpha=0.7, label=f\"{class_names[c]} ({auc_c:.3f})\")\n",
        "\n",
        "        # Plot Micro-Average ROC curve highlighted with bold black line\n",
        "        ax.plot(fpr_micro, tpr_micro, \"k-\", lw=2.5, label=f\"★ Micro-Avg ({auc_micro:.4f})\")\n",
        "        # Plot Random Chance diagonal line\n",
        "        ax.plot([0, 1], [0, 1], \"r--\", lw=1.2, alpha=0.7, label=\"Random Chance\")\n",
        "\n",
        "        ax.set_xlim([0.0, 1.0])\n",
        "        ax.set_ylim([0.0, 1.05])\n",
        "        ax.set_title(f\"ROC Feature: {model_name}\", fontweight=\"bold\", fontsize=10.5)\n",
        "        ax.set_xlabel(\"False Positive Rate (FPR)\", fontsize=8)\n",
        "        ax.set_ylabel(\"True Positive Rate (TPR)\", fontsize=8)\n",
        "        ax.grid(True, linestyle=\"--\", alpha=0.4)\n",
        "        ax.legend(fontsize=6, loc=\"lower right\", framealpha=0.85)\n",
        "\n",
        "    for idx in range(n_models, len(axes)):\n",
        "        axes[idx].axis(\"off\")\n",
        "\n",
        "    plt.suptitle(\"Individual Multi-Class OvR ROC & Micro-AUC Charts per Model Feature\", fontsize=14, fontweight=\"bold\", y=1.02)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "plot_individual_model_roc_curves(models_probs_dict, y_true_test, CIFAR10_CLASSES)\n",
        "\n",
        "# --- Per-Class Accuracy Grouped Bar Chart & Comparison Table ---\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
        "\n",
        "ax = axes[0]\n",
        "x = np.arange(len(CIFAR10_CLASSES))\n",
        "width = 0.12\n",
        "colors_list = [\"#1f77b4\", \"#ff7f0e\", \"#2ca02c\", \"#d62728\", \"#9467bd\", \"#8c564b\", \"#ffd700\"]\n",
        "for i, (name, res) in enumerate(eval_results.items()):\n",
        "    offset = (i - 3) * width\n",
        "    c = colors_list[i % len(colors_list)]\n",
        "    ax.bar(x + offset, res[\"per_class_acc\"], width, label=name, color=c, edgecolor=\"black\", linewidth=0.5)\n",
        "\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha=\"right\")\n",
        "ax.set_ylabel(\"Accuracy (%)\")\n",
        "ax.set_title(\"Per-Class Test Accuracy Comparison across Variants\", fontweight=\"bold\")\n",
        "ax.legend(fontsize=7)\n",
        "ax.grid(True, alpha=0.3, axis=\"y\")\n",
        "ax.set_ylim(0, 105)\n",
        "\n",
        "ax2 = axes[1]\n",
        "ax2.axis(\"off\")\n",
        "table_text = format_comparison_table(eval_results)\n",
        "ax2.text(0, 0.5, table_text, fontsize=8.5, family=\"monospace\", verticalalignment=\"center\", transform=ax2.transAxes)\n",
        "ax2.set_title(\"Benchmark Comparison Table\", fontsize=11, fontweight=\"bold\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# --- Normalized Confusion Matrix Heatmaps ---\n",
        "fig, axes = plt.subplots(2, 4, figsize=(18, 9))\n",
        "for idx, (name, res) in enumerate(eval_results.items()):\n",
        "    ax = axes.flatten()[idx]\n",
        "    cm = np.array(res[\"confusion_matrix\"])\n",
        "    cm_norm = cm / cm.sum(axis=1, keepdims=True)\n",
        "    im = ax.imshow(cm_norm, cmap=\"Blues\", vmin=0, vmax=1)\n",
        "    ax.set_xticks(range(len(CIFAR10_CLASSES)))\n",
        "    ax.set_yticks(range(len(CIFAR10_CLASSES)))\n",
        "    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha=\"right\", fontsize=6)\n",
        "    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=6)\n",
        "    thresh = cm_norm.max() / 2\n",
        "    for r in range(len(CIFAR10_CLASSES)):\n",
        "        for c in range(len(CIFAR10_CLASSES)):\n",
        "            ax.text(c, r, str(cm[r, c]), ha=\"center\", va=\"center\", fontsize=5.5,\n",
        "                    color=\"white\" if cm_norm[r, c] > thresh else \"black\")\n",
        "    ax.set_xlabel(\"Predicted\")\n",
        "    ax.set_ylabel(\"True\")\n",
        "    ax.set_title(name, fontsize=8.5, fontweight=\"bold\")\n",
        "\n",
        "for idx in range(len(eval_results), 8):\n",
        "    axes.flatten()[idx].axis(\"off\")\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# --- Chart 10: Misclassified Class Error Distribution & Top 10 Confused Class Pairs ---\n",
        "def plot_error_distribution(eval_results, class_names):\n",
        "    \"\"\"Plot Misclassified Class Error Distribution & Top 10 Confused Class Pairs.\"\"\"\n",
        "    best_name = max(eval_results, key=lambda k: eval_results[k][\"test_acc\"])\n",
        "    cm = np.array(eval_results[best_name][\"confusion_matrix\"])\n",
        "    n_classes = len(class_names)\n",
        "\n",
        "    total_per_class = cm.sum(axis=1)\n",
        "    correct_per_class = np.diag(cm)\n",
        "    errors_per_class = total_per_class - correct_per_class\n",
        "\n",
        "    confused_pairs = []\n",
        "    for r in range(n_classes):\n",
        "        for c in range(n_classes):\n",
        "            if r != c and cm[r, c] > 0:\n",
        "                pair_name = f\"{class_names[r]} → {class_names[c]}\"\n",
        "                confused_pairs.append((pair_name, cm[r, c]))\n",
        "    \n",
        "    confused_pairs.sort(key=lambda x: x[1], reverse=True)\n",
        "    top_pairs = confused_pairs[:10]\n",
        "\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "    # Subplot A: Per-Class Misclassification Error Count\n",
        "    ax1 = axes[0]\n",
        "    colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_classes))\n",
        "    bars = ax1.bar(class_names, errors_per_class, color=colors, edgecolor=\"black\", linewidth=0.8)\n",
        "    for bar in bars:\n",
        "        height = bar.get_height()\n",
        "        ax1.annotate(f'{int(height)}',\n",
        "                    xy=(bar.get_x() + bar.get_width() / 2, height),\n",
        "                    xytext=(0, 3),\n",
        "                    textcoords=\"offset points\",\n",
        "                    ha='center', va='bottom', fontsize=8, fontweight=\"bold\")\n",
        "\n",
        "    ax1.set_xticklabels(class_names, rotation=45, ha=\"right\")\n",
        "    ax1.set_ylabel(\"Number of Misclassified Samples (Errors)\")\n",
        "    ax1.set_title(f\"🏆 {best_name}: Per-Class Misclassification Error Count\", fontweight=\"bold\", fontsize=11)\n",
        "    ax1.grid(True, linestyle=\"--\", alpha=0.4, axis=\"y\")\n",
        "\n",
        "    # Subplot B: Top 10 Confused Class Pairs\n",
        "    ax2 = axes[1]\n",
        "    pair_names = [p[0] for p in top_pairs]\n",
        "    pair_counts = [p[1] for p in top_pairs]\n",
        "    colors_b = plt.cm.magma(np.linspace(0.4, 0.85, len(top_pairs)))\n",
        "\n",
        "    bars_b = ax2.barh(pair_names[::-1], pair_counts[::-1], color=colors_b[::-1], edgecolor=\"black\", linewidth=0.8)\n",
        "    for bar in bars_b:\n",
        "        width = bar.get_width()\n",
        "        ax2.annotate(f'{int(width)}',\n",
        "                    xy=(width, bar.get_y() + bar.get_height() / 2),\n",
        "                    xytext=(3, 0),\n",
        "                    textcoords=\"offset points\",\n",
        "                    ha='left', va='center', fontsize=8, fontweight=\"bold\")\n",
        "\n",
        "    ax2.set_xlabel(\"Number of Confusion Instances\")\n",
        "    ax2.set_title(f\"🏆 {best_name}: Top 10 Most Confused Class Pairs (True → Pred)\", fontweight=\"bold\", fontsize=11)\n",
        "    ax2.grid(True, linestyle=\"--\", alpha=0.4, axis=\"x\")\n",
        "\n",
        "    plt.suptitle(\"Error Analysis: Misclassified Class Distribution & Confusion Pairs\", fontsize=14, fontweight=\"bold\", y=1.02)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "plot_error_distribution(eval_results, CIFAR10_CLASSES)\n"
    ]
}
cells.append(cell_17)

# ==============================================================================
# CELL 18 (Markdown): Section 6.1
# ==============================================================================
cell_18 = {
    'cell_type': 'markdown',
    'id': 'cell_18',
    'metadata': {},
    'source': [
        "### 6.1 Visualize Predictions\n",
        "\n",
        "Display correctly classified and misclassified test-set examples for the top-performing model variant. Images are denormalized from ImageNet statistics using:\n",
        "\n",
        "$$\\mathbf{x}_{\\text{display}} = \\mathbf{x}_{\\text{norm}} \\cdot \\boldsymbol{\\sigma} + \\boldsymbol{\\mu}$$\n"
    ]
}
cells.append(cell_18)

# ==============================================================================
# CELL 19 (Code): Prediction Visualization Grids
# ==============================================================================
cell_19 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_19',
    'metadata': {},
    'outputs': [],
    'source': [
        "# --- Helper: reverse ImageNet normalization for display ---\n",
        "def denormalize(tensor, mean, std):\n",
        "    \"\"\"Denormalize ImageNet tensor back to [0, 1] range for image rendering.\"\"\"\n",
        "    mean = torch.tensor(mean).view(3, 1, 1)\n",
        "    std = torch.tensor(std).view(3, 1, 1)\n",
        "    return tensor * std + mean\n",
        "\n",
        "if eval_results:\n",
        "    best_name = max(eval_results, key=lambda k: eval_results[k][\"test_acc\"])\n",
        "    print(f\"Top performing variant for visualization: {best_name} (test acc={eval_results[best_name]['test_acc']:.2f}%)\")\n",
        "\n",
        "    best_model = rn_sota\n",
        "    best_model.eval()\n",
        "    correct_examples, incorrect_examples = [], []\n",
        "    N_CORRECT, N_INCORRECT = 8, 8\n\n    with torch.no_grad():\n",
        "        for images, labels in test_loader:\n",
        "            images_dev, labels_dev = images.to(device), labels.to(device)\n",
        "            outputs = best_model(images_dev)\n",
        "            probs = torch.softmax(outputs, dim=1)\n",
        "            confs, preds = probs.max(1)\n",
        "\n",
        "            for i in range(len(images)):\n",
        "                img_denorm = denormalize(images[i].cpu(), IMAGENET_MEAN, IMAGENET_STD)\n",
        "                conf_pct = confs[i].item() * 100.0\n",
        "                if len(correct_examples) < N_CORRECT and preds[i] == labels_dev[i]:\n",
        "                    correct_examples.append((img_denorm, preds[i].item(), labels[i].item(), conf_pct))\n",
        "                elif len(incorrect_examples) < N_INCORRECT and preds[i] != labels_dev[i]:\n",
        "                    incorrect_examples.append((img_denorm, preds[i].item(), labels[i].item(), conf_pct))\n",
        "                if len(correct_examples) >= N_CORRECT and len(incorrect_examples) >= N_INCORRECT:\n",
        "                    break\n",
        "            if len(correct_examples) >= N_CORRECT and len(incorrect_examples) >= N_INCORRECT:\n",
        "                break\n",
        "\n",
        "    # --- Grid 1: Correct predictions ---\n",
        "    fig, axes = plt.subplots(2, 4, figsize=(14, 6))\n",
        "    for idx, (img, pred, true, conf) in enumerate(correct_examples):\n",
        "        ax = axes.flatten()[idx]\n",
        "        img = torch.clamp(img, 0, 1)\n",
        "        ax.imshow(img.permute(1, 2, 0).numpy())\n",
        "        ax.set_title(f\"Classified: {CIFAR10_CLASSES[pred]}\\n({conf:.1f}% conf)\", fontsize=9, fontweight=\"bold\", color=\"green\")\n",
        "        ax.axis(\"off\")\n",
        "    fig.suptitle(f\"{best_name}: Correctly Classified Test Examples\", fontsize=12, fontweight=\"bold\", y=1.02)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "    # --- Grid 2: Incorrect predictions ---\n",
        "    fig, axes = plt.subplots(2, 4, figsize=(14, 6))\n",
        "    for idx, (img, pred, true, conf) in enumerate(incorrect_examples):\n",
        "        ax = axes.flatten()[idx]\n",
        "        img = torch.clamp(img, 0, 1)\n",
        "        ax.imshow(img.permute(1, 2, 0).numpy())\n",
        "        ax.set_title(f\"Pred: {CIFAR10_CLASSES[pred]} ({conf:.1f}%)\\nTrue: {CIFAR10_CLASSES[true]}\", fontsize=8, color=\"red\")\n",
        "        ax.axis(\"off\")\n",
        "    fig.suptitle(f\"{best_name}: Misclassified Test Examples\", fontsize=12, fontweight=\"bold\", y=1.02)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "    print(f\"✓ Visualization grids generated: {len(correct_examples)} correct, {len(incorrect_examples)} misclassified\")\n"
    ]
}
cells.append(cell_19)

# ==============================================================================
# CELL 20 (Markdown): Section 7
# ==============================================================================
cell_20 = {
    'cell_type': 'markdown',
    'id': 'cell_20',
    'metadata': {},
    'source': [
        "## 7. Save Models & Results\n",
        "\n",
        "Best model checkpoints are persisted to `experiments/checkpoints/<run_name>_best.pt`. Here we save the benchmark comparison text table and JSON test metrics for reporting.\n"
    ]
}
cells.append(cell_20)

# ==============================================================================
# CELL 21 (Code): Results Persistence
# ==============================================================================
cell_21 = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'cell_21',
    'metadata': {},
    'outputs': [],
    'source': [
        "if eval_results:\n",
        "    results_dir = PROJECT_ROOT / \"experiments\" / \"results\"\n",
        "    os.makedirs(results_dir, exist_ok=True)\n",
        "    \n",
        "    table_str = format_comparison_table(eval_results)\n",
        "    (results_dir / \"comparison_table.txt\").write_text(table_str + \"\\n\")\n",
        "    \n",
        "    summary_dict = {\n",
        "        name: {\n",
        "            \"mode\": res[\"mode\"],\n",
        "            \"test_loss\": round(res[\"test_loss\"], 4),\n",
        "            \"test_acc\": round(res[\"test_acc\"], 2),\n",
        "            \"per_class_acc\": [round(v, 2) for v in res[\"per_class_acc\"]],\n",
        "        }\n",
        "        for name, res in eval_results.items()\n",
        "    }\n",
        "    (results_dir / \"test_metrics.json\").write_text(json.dumps(summary_dict, indent=2) + \"\\n\")\n",
        "    \n",
        "    history_dict = {\n",
        "        name: {\n",
        "            \"train_losses\": [round(v, 4) for v in res[\"train_losses\"]],\n",
        "            \"val_losses\": [round(v, 4) for v in res[\"val_losses\"]],\n",
        "            \"val_accs\": [round(v, 2) for v in res[\"val_accs\"]],\n",
        "            \"best_epoch\": res[\"best_epoch\"],\n",
        "            \"best_val_acc\": round(res[\"best_val_acc\"], 2),\n",
        "        }\n",
        "        for name, res in training_results.items()\n",
        "    }\n",
        "    (results_dir / \"training_history.json\").write_text(json.dumps(history_dict, indent=2) + \"\\n\")\n",
        "    print(f\"✓ All evaluation results & training history successfully saved to {results_dir}\")\n",
        "else:\n",
        "    print(\"No evaluation results — nothing to save.\")\n"
    ]
}
cells.append(cell_21)

nb_data = {
    'cells': cells,
    'metadata': {
        'language_info': {
            'name': 'python'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

os.makedirs('scratch', exist_ok=True)
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb_data, f, indent=1)

print(f"Successfully generated upgraded notebook at {notebook_path} with {len(cells)} cells.")
