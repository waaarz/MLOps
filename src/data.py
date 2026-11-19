from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.reproducibility import make_generator, seed_worker


def base_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def train_transform(enabled: bool, horizontal_flip: float, rotation_degrees: float) -> transforms.Compose:
    steps = []
    if enabled:
        steps.extend([
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            transforms.RandomRotation(degrees=rotation_degrees),
        ])
    steps.extend(base_transform().transforms)
    return transforms.Compose(steps)


def build_training_loaders(processed_dir: Path, params: dict):
    seed = int(params["train"]["random_state"])
    batch_size = int(params["train"]["batch_size"])
    aug = params["augmentation"]

    train_ds = datasets.ImageFolder(
        processed_dir / "train",
        transform=train_transform(
            bool(aug["enabled"]),
            float(aug["horizontal_flip"]),
            float(aug["rotation_degrees"]),
        ),
    )
    val_ds = datasets.ImageFolder(processed_dir / "validation", transform=base_transform())

    if train_ds.classes != val_ds.classes:
        raise RuntimeError("Class ordering differs between train and validation splits")

    generator = make_generator(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, train_ds.classes


def build_test_loader(processed_dir: Path, batch_size: int):
    test_ds = datasets.ImageFolder(processed_dir / "test", transform=base_transform())
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, test_ds.classes
