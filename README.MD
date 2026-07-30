# Deep Learning Course - LAB2

## Project Overview

This project implements a modular, production-ready data pipeline for computer vision tasks, specifically designed for CIFAR-10 classification. The architecture follows the Separation of Concerns (SoC) principle, providing a clean separation between data processing, model architecture, training, and evaluation components.

## Architecture

The project is organized into distinct layers:

```
Dataset → Inspection → Statistics → Transforms → DataLoader → Models → Training → Evaluation
```

Each layer exposes a clean public API and does not depend on implementation details of other layers. This design ensures modularity, reusability, and extensibility for future Computer Vision projects beyond CIFAR-10.

## Folder Structure

```
Deeplearning_Course_UTH/
│
├── configs/
│   └── data.yaml              # Centralized configuration for data pipeline
│
├── data/                     # Primary Data Pipeline (Single Source of Truth)
│   ├── external/             # Raw dataset storage
│   │   └── CIFAR-10/
│   ├── processed/            # Processed data and statistics
│   │   ├── dataset_statistics.json
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── dataset.py            # Dataset download and loading
│   ├── inspection.py         # Data validation and quality checks
│   ├── statistics.py         # Dataset statistics computation
│   ├── transforms.py         # Transform pipelines
│   ├── dataloader.py         # DataLoader creation
│   └── config.py             # Configuration loader
│
├── docs/                     # Documentation
│   ├── dataset.md
│   ├── inspection.md
│   ├── statistics.md
│   ├── transforms.md
│   └── dataloader.md
│
├── src/                      # Deep Learning Components
│   ├── models/               # Model architectures (ResNet, DenseNet, etc.)
│   ├── training/             # Training logic (loops, optimizers, schedulers)
│   ├── evaluation/           # Evaluation metrics and testing
│   ├── utils/                # Shared utilities
│   └── data/                 # Legacy (Deprecated - Do Not Extend)
│
├── tests/                    # Unit tests
│   ├── test_dataset.py
│   ├── test_statistics.py
│   ├── test_transforms.py
│   └── test_dataloader.py
│
└── README.md
```

## Data Pipeline Workflow

1. **Dataset** (`data/dataset.py`)
   - Download CIFAR-10 dataset
   - Load training and test sets
   - Provide dataset metadata

2. **Inspection** (`data/inspection.py`)
   - Validate data integrity
   - Check class distribution
   - Verify label consistency
   - Detect corrupted images

3. **Statistics** (`data/statistics.py`)
   - Compute mean and standard deviation
   - Persist statistics for reuse
   - Provide normalization parameters

4. **Transforms** (`data/transforms.py`)
   - Create training transform pipeline with augmentation
   - Create evaluation transform pipeline
   - Support predefined configurations

5. **DataLoader** (`data/dataloader.py`)
   - Create train/val/test DataLoaders
   - Maintain persistent train/val/test split
   - Configure batch processing options

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

3. Install additional dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

Key dependencies:
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities and pretrained models
- `pyyaml` - Configuration file parsing
- `pytest` - Unit testing framework

See `requirements.txt` for complete list.

## Quick Start

### Basic Usage

```python
from data import download_cifar10, get_cifar10_loaders

# Download dataset
download_cifar10()

# Create DataLoaders
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)

# Iterate through batches
for images, labels in train_loader:
    # Your training code here
    pass
```

### Advanced Usage

```python
from data import (
    download_cifar10,
    load_cifar10_dataset,
    get_train_transform,
    get_eval_transform,
    get_single_loader,
    compute_dataset_statistics,
    save_statistics,
)

# Download dataset
download_cifar10()

# Load dataset
train_set = load_cifar10_dataset(train=True)

# Compute statistics
mean, std = compute_dataset_statistics(train_set)
save_statistics(mean, std, "data/processed/dataset_statistics.json")

# Create custom transforms
train_transform = get_train_transform(mean=mean.tolist(), std=std.tolist())
eval_transform = get_eval_transform(mean=mean.tolist(), std=std.tolist())

# Load dataset with transforms
train_set = load_cifar10_dataset(train=True, transform=train_transform)

# Create DataLoader
train_loader = get_single_loader(train_set, batch_size=64, shuffle=True)
```

## Configuration

The data pipeline uses a centralized YAML configuration file at `configs/data.yaml`.

### Configuration Sections

**Dataset:**
- `dataset.name`: Dataset name
- `dataset.root`: Dataset storage path
- `dataset.download`: Auto-download flag

**Normalization:**
- `normalization.imagenet`: ImageNet statistics for pretrained models
- `normalization.cifar10`: CIFAR-10 statistics

**Transforms:**
- `transforms.train`: Training transform parameters
- `transforms.eval`: Evaluation transform parameters

**DataLoader:**
- `dataloader.batch_size`: Batch size
- `dataloader.num_workers`: Number of worker processes
- `dataloader.pin_memory`: GPU memory pinning
- `dataloader.persistent_workers`: Keep workers alive

**Training:**
- `training.seed`: Random seed
- `training.device`: Device selection
- `training.deterministic`: Deterministic mode

### Using Configuration

```python
from data.config import load_config, get_dataloader_config

config = load_config()
dataloader_config = get_dataloader_config(config)
```

## Running the Data Pipeline

### Complete Pipeline Example

```python
import logging
from data import (
    download_cifar10,
    load_cifar10_dataset,
    get_train_transform,
    get_eval_transform,
    get_cifar10_loaders,
    count_images,
    check_distribution,
    verify_labels,
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Step 1: Download dataset
download_cifar10()

# Step 2: Load dataset
train_set = load_cifar10_dataset(train=True)

# Step 3: Inspect dataset
print(f"Total images: {count_images(train_set)}")
print(f"Distribution: {check_distribution(train_set)}")
print(f"Label verification: {verify_labels(train_set)}")

# Step 4: Create DataLoaders
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)

# Step 5: Use in training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Training code
        pass
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_dataset.py

# Run with verbose output
pytest tests/ -v
```

## TorchVision Policy

TorchVision is encapsulated within the data pipeline modules. Training code should import reusable interfaces from the `data/` package instead of calling TorchVision directly.

**Allowed:**
```python
from data.dataset import load_cifar10_dataset
from data.transforms import get_train_transform
from data.dataloader import get_cifar10_loaders
```

**Not allowed in training scripts:**
```python
from torchvision.datasets import CIFAR10
from torchvision import transforms
```

## Future Extensions

The modular architecture supports easy extension to:

- **New Datasets**: Add dataset-specific loaders in `data/dataset.py`
- **New Transforms**: Extend `data/transforms.py` with custom augmentations
- **New Models**: Add architectures in `src/models/`
- **New Metrics**: Implement evaluation metrics in `src/evaluation/`
- **New Utilities**: Add helper functions in `src/utils/`

The separation of concerns ensures that changes in one layer do not affect others, making the codebase maintainable and extensible.

## Legacy Code

The `src/data/` directory contains legacy code for backward compatibility only. It is marked as deprecated and should not be extended. All new data pipeline development must occur in the `data/` directory.

## License

This project is part of a Deep Learning course.
