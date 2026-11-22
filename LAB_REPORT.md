# Laboratory Report

| File / setting | Problem | Suggested fix |
|---|---|---|
| `src/reproducibility.py` - `make_generator()` | The function returns an unseeded `torch.Generator`, even though a requested seed is passed. The test reports `g.initial_seed()` as a random large value instead of `7`. | Return `torch.Generator().manual_seed(seed)`. |
| `src/reproducibility.py` - `seed_worker()` | Worker seeding does not derive NumPy and Python seeds from PyTorch's DataLoader worker seed. The test derives a mismatch between the expected and observed values. | Use `torch.initial_seed() % (2**32)` in `seed_worker`, then seed both `random` and `numpy.random`. |
| `src/validate_data.py` - `validate_dataset()` | The core data-quality gate is unimplemented and raises `NotImplementedError`. Missing splits, extra classes, empty classes, wrong counts, and corrupt images therefore cannot be rejected. | Implement all required structural, count, and Pillow decode checks and return a deterministic JSON report with a clear `PASS` or `FAIL` status. |
| `src/provenance.py` - `build_run_manifest()` | Run provenance is unimplemented, so Git identity, environment versions, data identity, device, seed, and configuration cannot be captured. | Implement the documented manifest fields and handle unavailable Git/DVC/CUDA metadata gracefully with explicit values. |
| `src/train.py` - TODO 1-7 | The training Run does not capture provenance, parameters, dataset input, epoch metrics, timing, artifacts, or the MLflow PyTorch model. This also prevents controlled run selection and test handoff. | Complete the TODO blocks in order, first creating `run_manifest.json`, then logging parameters, data, metrics, artifacts, and a CPU-evaluation model without test data. |
| MLflow Run history / `params.yaml` - `selection.run_id` | No completed, tracked training Run or selected Run is present, and `selection.run_id` is empty. Final evaluation is therefore correctly blocked. | Run controlled candidate training, compare validation results, then set the chosen FINISHED Run ID only after confirming data identity and image size. |
| DVC pipeline / project root | No `dvc.yaml` is present, and `data/raw` plus prepared manifests are absent. The DVC stage dependency and quality gate are not operational. | Add and run a reproducible DVC DAG for download/check, validate, prepare, train, and evaluate; make PREPARE depend on the successful quality report. |
| Git repository state | The repository has no commits (`git log` fails because `main` has no commits), and every project file is untracked. A provenance SHA and dirty/clean state cannot be recorded meaningfully. | Make the initial implementation commit, then commit each milestone. |
| Test-set use / `src/evaluate.py` | This surface is protected rather than currently leaking: training uses train/validation only, and evaluation requires explicit human Run selection plus FINISHED/status, experiment, size, and data-identity checks. | Preserve these checks and add a test that evaluation never runs while candidate training or hyperparameter selection is ongoing. |
| Dataset / local environment | Raw and processed data are not currently restored; only the three archive ZIPs and `.gitkeep` files exist under `data/`, `models/`, and `reports/`. | Acquire/extract data, pass the quality gate, and run PREPARE before training. Use the lock file and `.venv` for environment replay. |

## Python environment

- `pyproject.toml` declares the project metadata, supported Python range (`>=3.11,<3.13`), runtime dependencies, development dependencies, and the CPU-only PyTorch package index. It is the human-maintained dependency specification.
- `uv.lock` records the resolved dependency graph and exact package versions for reproducible installs. Recreate the environment with `uv sync --locked` rather than changing dependencies ad hoc.
- `.venv/` is the project-local virtual environment containing the installed Python executable and packages. It isolates this project from system Python and other projects.

## Engineering checkpoint 1

**Why is a working `model.pt` insufficient evidence that an experiment is reproducible?**

A working `model.pt` proves only that serialized tensors can be loaded and used. It does not prove that another execution can reconstruct the same model or results.

It does not record which code produced the weights (Git commit), which data they were trained on (data version/hash), which package versions and hardware were used, which hyper-parameters and seed were set, or how the model was scored on which split. Without that information nobody can re-run the experiment, check that they obtain the same result, or even tell two similar-looking `model.pt` files apart. Reproducibility requires the *recipe* (code + data + environment + settings + seeds) to be recorded and versioned, not only the *product*.

## Milestone 1 - Repeatable training

Test status before the fix (`uv run --locked pytest -q`): 10 failed / 9 passed. After completing `src/reproducibility.py`, `tests/test_reproducibility.py` gives `3 passed`. The remaining failures belong to the validator and provenance TODOs (later milestones).

What was fixed in `src/reproducibility.py`:

- `set_reproducibility()` seeds Python `random`, NumPy, PyTorch CPU, PyTorch CUDA (only when available) and puts cuDNN in deterministic mode (`benchmark = False`). The starter version called `torch.cuda.manual_seed_all` unconditionally and duplicated calls.
- `make_generator(seed)` returns `torch.Generator().manual_seed(seed)` so that `DataLoader` shuffling is repeatable.
- `seed_worker()` derives the NumPy / Python seed from `torch.initial_seed() % 2**32` (the seed PyTorch gives each worker), instead of drawing a fresh random number.

**Engineering checkpoint 2 - does `random_state=42` guarantee bit-for-bit identical results everywhere?**
No. A seed only fixes the pseudo-random streams. Results can still differ across hardware (CPU vs GPU, different CPU instruction sets / BLAS or oneDNN kernels), across library versions (PyTorch, CUDA, cuDNN), across operating systems, and because some GPU kernels are non-deterministic (atomic adds) or floating-point sums are not associative when the order of the operations changes (threads, batch size). So a seed gives *repeatability on the same stack*, and the stack therefore has to be recorded too (`uv.lock`, run manifest).
