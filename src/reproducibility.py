from __future__ import annotations

import random

import numpy as np
import torch


def set_reproducibility(seed: int) -> None:
    """Seed the relevant random number generators.

    Python, NumPy, PyTorch (CPU) and PyTorch CUDA (when available) are seeded,
    and cuDNN is switched to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded torch.Generator used for deterministic shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
