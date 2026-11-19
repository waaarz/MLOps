import random

import numpy as np
import torch

from src.reproducibility import make_generator, seed_worker, set_reproducibility


def sample_triplet():
    return random.random(), float(np.random.rand()), float(torch.rand(1).item())


def test_global_seeding_repeats_python_numpy_torch():
    set_reproducibility(123)
    a = sample_triplet()
    set_reproducibility(123)
    b = sample_triplet()
    assert a == b


def test_generator_uses_requested_seed():
    g = make_generator(7)
    assert g.initial_seed() == 7
    g1 = make_generator(7)
    g2 = make_generator(7)
    assert torch.randperm(20, generator=g1).tolist() == torch.randperm(20, generator=g2).tolist()


def test_seed_worker_controls_numpy_and_python(monkeypatch):
    seed = 12345
    monkeypatch.setattr(torch, "initial_seed", lambda: seed)
    seed_worker(0)
    observed = (random.random(), float(np.random.rand()))

    random.seed(seed % (2**32))
    np.random.seed(seed % (2**32))
    expected = (random.random(), float(np.random.rand()))
    assert observed == expected
