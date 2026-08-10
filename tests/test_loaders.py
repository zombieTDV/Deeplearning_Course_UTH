"""test_loaders.py — get_cifar10_loaders structure + config defaults (TST-2, CQ-2, PERF-3)."""


from src.data.dataloader import _ApplyTransform, _config_defaults, get_cifar10_loaders
from tests.conftest import requires_data


def test_config_defaults_wired_from_yaml():
    """configs/data.yaml is the single source of defaults (audit CQ-2)."""
    cfg = _config_defaults()
    assert cfg["batch_size"] == 64
    assert cfg["num_workers"] == 0          # safe default on Python 3.14
    assert cfg["data_root"].endswith("data" + __import__("os").sep + "raw")


@requires_data
def test_get_cifar10_loaders_counts_and_shapes():
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=8, num_workers=0, data_root=str(__import__("pathlib").Path(
            __file__).resolve().parents[1] / "data" / "raw"))
    assert len(train_loader.dataset) == 45_000
    assert len(val_loader.dataset) == 5_000
    assert len(test_loader.dataset) == 10_000
    images, labels = next(iter(train_loader))
    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)


@requires_data
def test_val_and_train_share_raw_dataset():
    """PERF-3: train+val share one raw train instance (no triple in-memory copy)."""
    tr, val, _ = get_cifar10_loaders(batch_size=8, num_workers=0,
                                     data_root=str(__import__("pathlib").Path(
                                         __file__).resolve().parents[1] / "data" / "raw"))
    train_raw = tr.dataset.dataset.dataset          # Subset -> _ApplyTransform -> raw CIFAR10
    val_raw = val.dataset.dataset.dataset
    assert train_raw is val_raw, "train and val must share the raw train dataset"


def test_apply_transform_wrapper():
    import numpy as np
    import torchvision.transforms as T

    class DS:
        def __len__(self):
            return 3

        def __getitem__(self, i):
            return np.zeros((8, 8, 3), dtype=np.uint8), i  # PIL-like RGB image

    wrapped = _ApplyTransform(DS(), T.ToTensor())
    img, label = wrapped[0]
    assert tuple(img.shape) == (3, 8, 8) and label == 0
