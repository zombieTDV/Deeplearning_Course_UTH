import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import DataLoader, Subset, ConcatDataset


def get_fashionmnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def load_fashionmnist(transform, root='../data'):
    train_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def create_full_split(train_size=50000, val_size=6000, test_size=14000,
                      split_dir='../splits', seed=42, root='../data'):
    split_dir = Path(split_dir)
    split_dir.mkdir(exist_ok=True)

    train_file = split_dir / 'train_indices.json'
    val_file = split_dir / 'val_indices.json'
    test_file = split_dir / 'test_indices.json'

    if train_file.exists() and val_file.exists() and test_file.exists():
        print(f'[data_utils] All split files exist — loading from {split_dir}')
        return

    root_path = Path(root)
    full_dataset = ConcatDataset([
        torchvision.datasets.FashionMNIST(root=str(root_path), train=True, download=True,
                                          transform=transforms.ToTensor()),
        torchvision.datasets.FashionMNIST(root=str(root_path), train=False, download=True,
                                          transform=transforms.ToTensor()),
    ])
    total = len(full_dataset)
    assert train_size + val_size + test_size == total, \
        f'{train_size} + {val_size} + {test_size} = {train_size+val_size+test_size} != {total}'

    indices = np.arange(total)
    np.random.seed(seed)
    np.random.shuffle(indices)

    json.dump(indices[:train_size].tolist(), open(train_file, 'w'))
    json.dump(indices[train_size:train_size+val_size].tolist(), open(val_file, 'w'))
    json.dump(indices[train_size+val_size:].tolist(), open(test_file, 'w'))
    print(f'[data_utils] Split created: train={train_size}, val={val_size}, test={test_size}')
    print(f'[data_utils] Saved to {split_dir}/')


def _load_one(file, full_dataset):
    with open(file) as f:
        return Subset(full_dataset, json.load(f))


def load_subsets(batch_size=64, val_batch_size=256, split_dir='../splits', root='../data'):
    split_dir = Path(split_dir)
    train_file = split_dir / 'train_indices.json'
    val_file = split_dir / 'val_indices.json'
    test_file = split_dir / 'test_indices.json'

    for f in [train_file, val_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(
                f'Missing split file: {f}. Run create_full_split() first.')

    tf = get_fashionmnist_transforms()
    root_path = Path(root)
    full_dataset = ConcatDataset([
        torchvision.datasets.FashionMNIST(root=str(root_path), train=True, download=False,
                                          transform=tf),
        torchvision.datasets.FashionMNIST(root=str(root_path), train=False, download=False,
                                          transform=tf),
    ])

    train_subset = _load_one(train_file, full_dataset)
    val_subset = _load_one(val_file, full_dataset)
    test_subset = _load_one(test_file, full_dataset)

    class_names = torchvision.datasets.FashionMNIST(
        root=str(root_path), train=True, download=False).classes

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names
