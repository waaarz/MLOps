from __future__ import annotations

import random
import numpy as np
import torch


def set_reproducibility(seed: int) -> None:
    """Seed the relevant random number generators.

    TODO: complete this function so that Python, NumPy and PyTorch are
    controlled. Improve CUDA/cuDNN repeatability when CUDA is available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # TODO: NumPy
    np.random.seed(seed)
    # TODO: PyTorch CPU
    torch.manual_seed(seed)
    # TODO: CUDA, if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # TODO: cuDNN deterministic / benchmark flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed."""
    # TODO: implement worker seeding for NumPy and Python random.
    np.random.seed(random.randint(0, 2**32 - 1))
    random.seed(random.randint(0, 2**32 - 1))
    pass


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded torch.Generator used for deterministic shuffling."""
    # TODO
    return torch.Generator()
