"""CIFAR-10 with symmetric or asymmetric label noise.

Split strategy
--------------
1. Take val_per_class clean examples per class from the training pool → validation set.
2. Apply noise only to the remaining training examples.
3. Test set is always clean (standard CIFAR-10 test split).

Asymmetric noise map (CIFAR-10, matches literature convention):
  airplane(0)→bird(2), automobile(1)→truck(9),
  bird(2)→airplane(0), truck(9)→automobile(1),
  cat(3)→dog(5), dog(5)→cat(3),
  deer(4)→horse(7), horse(7)→deer(4).
  (Remaining classes are left untouched.)
"""
import copy
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

ASYM_MAP = {
    0: 2,   # airplane → bird
    1: 9,   # automobile → truck
    2: 0,   # bird → airplane
    9: 1,   # truck → automobile
    3: 5,   # cat → dog
    5: 3,   # dog → cat
    4: 7,   # deer → horse
    7: 4,   # horse → deer
}


# ------------------------------------------------------------------
# Dataset wrapper with overridable labels
# ------------------------------------------------------------------
class NoisyCIFAR10(Dataset):
    """CIFAR-10 subset with potentially corrupted labels.

    Parameters
    ----------
    base_dataset : torchvision Dataset
        The underlying (clean) CIFAR-10 dataset (train split).
    indices : list[int]
        Which samples to include.
    labels : list[int]
        Possibly-noisy labels, one per index.
    transform : callable, optional
    """

    def __init__(self, base_dataset, indices: List[int], labels: List[int], transform=None):
        self.data = base_dataset.data[indices]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        # data is numpy uint8 HWC; transforms expect PIL or tensor
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def _raw_cifar10(data_root: str):
    """Read CIFAR-10 batch files directly, bypassing torchvision's download/MD5 checks.

    Used as a fallback when torchvision's download fails (e.g. Colab blocks the URL).
    Expects the data to have been seeded first via seed_cifar10.py.
    Returns a pair of SimpleNamespace objects with .data (N,32,32,3 uint8) and .targets (list[int]).
    """
    import os
    import pickle
    from types import SimpleNamespace

    base = os.path.join(data_root, 'cifar-10-batches-py')
    if not os.path.isdir(base):
        raise RuntimeError(
            f"CIFAR-10 not found at {base} and automatic download failed.\n"
            "On Colab, seed the data first:  python seed_cifar10.py --data_root ./data"
        )

    chunks, train_labels = [], []
    for i in range(1, 6):
        with open(os.path.join(base, f'data_batch_{i}'), 'rb') as f:
            e = pickle.load(f, encoding='latin1')
        chunks.append(e['data'])
        train_labels.extend(e.get('labels', e.get('fine_labels', [])))
    train_data = np.concatenate(chunks).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    with open(os.path.join(base, 'test_batch'), 'rb') as f:
        te = pickle.load(f, encoding='latin1')
    test_data   = te['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = te.get('labels', te.get('fine_labels', []))

    return (SimpleNamespace(data=train_data, targets=train_labels),
            SimpleNamespace(data=test_data,  targets=test_labels))


def build_cifar10_datasets(
    data_root: str = "./data",
    noise_rate: float = 0.4,
    noise_type: str = "uniform",
    val_size: int = 1000,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Return (train_dataset, val_dataset, test_dataset).

    val_dataset is a guaranteed clean balanced set of val_size examples
    (val_size // 10 per class). train_dataset has noisy labels on the
    remainder.
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    val_per_class = val_size // 10

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    try:
        raw_train = datasets.CIFAR10(data_root, train=True,  download=True)
        raw_test  = datasets.CIFAR10(data_root, train=False, download=True, transform=eval_transform)
    except Exception:
        raw_train_ns, raw_test_ns = _raw_cifar10(data_root)
        raw_train = raw_train_ns
        raw_test  = NoisyCIFAR10(raw_test_ns, list(range(len(raw_test_ns.targets))),
                                 raw_test_ns.targets, transform=eval_transform)

    targets = np.array(raw_train.targets)

    # Step 1: carve out clean balanced validation indices
    val_indices: List[int] = []
    for c in range(10):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        val_indices.extend(class_idx[:val_per_class].tolist())

    val_set = set(val_indices)
    train_indices = [i for i in range(len(targets)) if i not in val_set]

    # Step 2: build noisy labels for training split
    train_labels = _apply_noise(
        targets[train_indices].tolist(), noise_rate, noise_type, rng
    )

    # Step 3: build clean labels for val split
    val_labels = targets[val_indices].tolist()

    train_dataset = NoisyCIFAR10(raw_train, train_indices, train_labels, transform=train_transform)
    val_dataset   = NoisyCIFAR10(raw_train, val_indices,   val_labels,   transform=eval_transform)

    return train_dataset, val_dataset, raw_test


def _apply_noise(
    labels: List[int],
    noise_rate: float,
    noise_type: str,
    rng: np.random.RandomState,
) -> List[int]:
    labels = list(labels)
    n = len(labels)
    if noise_type == "uniform":
        # Each label flipped to a uniformly random label (may stay same)
        flip_mask = rng.rand(n) < noise_rate
        for i in range(n):
            if flip_mask[i]:
                labels[i] = int(rng.randint(0, 10))
    elif noise_type == "asymmetric":
        flip_mask = rng.rand(n) < noise_rate
        for i in range(n):
            if flip_mask[i]:
                labels[i] = ASYM_MAP.get(labels[i], labels[i])
    else:
        raise ValueError(f"Unknown noise_type={noise_type!r}. Choose 'uniform' or 'asymmetric'.")
    return labels
