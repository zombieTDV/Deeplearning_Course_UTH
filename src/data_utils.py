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


def get_augmented_transforms():
    return transforms.Compose([
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=0),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def load_fashionmnist(transform, root='../data', train_transform=None):
    train_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True,
        transform=train_transform if train_transform else transform
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


def load_subsets(batch_size=64, val_batch_size=256, split_dir='../splits', root='../data',
                 train_transform=None):
    split_dir = Path(split_dir)
    train_file = split_dir / 'train_indices.json'
    val_file = split_dir / 'val_indices.json'
    test_file = split_dir / 'test_indices.json'

    for f in [train_file, val_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(
                f'Missing split file: {f}. Run create_full_split() first.')

    tf_plain = get_fashionmnist_transforms()
    tf_train = train_transform if train_transform else tf_plain
    root_path = Path(root)

    full_train = ConcatDataset([
        torchvision.datasets.FashionMNIST(root=str(root_path), train=True, download=False,
                                          transform=tf_train),
        torchvision.datasets.FashionMNIST(root=str(root_path), train=False, download=False,
                                          transform=tf_train),
    ])
    full_test = ConcatDataset([
        torchvision.datasets.FashionMNIST(root=str(root_path), train=True, download=False,
                                          transform=tf_plain),
        torchvision.datasets.FashionMNIST(root=str(root_path), train=False, download=False,
                                          transform=tf_plain),
    ])

    train_subset = _load_one(train_file, full_train)
    val_subset = _load_one(val_file, full_train)
    test_subset = _load_one(test_file, full_test)

    class_names = torchvision.datasets.FashionMNIST(
        root=str(root_path), train=True, download=False).classes

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names
