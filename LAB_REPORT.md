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

## Milestone 2 - Extract and track the images

- Option A (included archives): `download_dataset.py --archive-dir dataset-archives` verified the three SHA-256 checksums and extracted the images into `data/raw/`.
- `check_dataset.py`: train 1034 (345 / 348 / 341), validation 133 (44 / 45 / 44), test 128 (43 / 43 / 42) - all counts match the course archives.
- `dvc init`, `dvc add data/raw`, `dvc remote add -d lab-data ../ai53-phase1-dvc-storage`. `data/raw.dvc` is the small pointer Git tracks (directory hash `2b62d5a2...dir`, 179.9 MB).

**Engineering checkpoint 3**

- The **Git remote** (GitHub) stores the code, configuration, history and the small pointer files (`dvc.yaml`, `dvc.lock`, `data/raw.dvc`) - not the heavy data.
- The **DVC remote** (`lab-data`, the folder `../ai53-phase1-dvc-storage`) stores the actual content of the DVC-tracked files (raw images, and later processed data and model files), addressed by content hash.
- `git push` does **not** transfer the DVC-managed raw images; they are ignored by Git, and only `dvc push` sends them to the DVC remote.

## Milestone 3 - Data-quality gate

`validate_dataset()` in `src/validate_data.py` now: (1) raises `FileNotFoundError` if a split folder is missing; (2) raises `RuntimeError` if the class folders differ from the three expected classes; (3) raises if a split's image count differs from the course archives (1034 / 133 / 128); (4) opens every image and calls `Image.load()` so the pixels are really decoded (a truncated/corrupt file raises Pillow's `OSError` / `UnidentifiedImageError`); (5) raises if a class has no image. Files are listed in sorted order and the report has no timestamp or absolute path, so two runs on the same data give byte-identical JSON (checked with `md5sum`: identical hashes).

Results: `pytest tests/test_data_validation.py tests/test_validation_contract.py` -> `7 passed`; `python -m src.validate_data` -> `"status": "PASS"`, `total_images: 1295`, written to `reports/data_quality.json`.

## Milestone 4 - DVC pipeline (`dvc.yaml`)

Four stages (`validate -> prepare -> train` and `prepare -> evaluate`). `dvc stage list` shows all four and `dvc dag` confirms that `evaluate` depends on `prepare` only, not on `train`:

```
data/raw.dvc -> validate -> prepare -> train
                  data/raw.dvc -> prepare -> evaluate
```

Answers to the design questions:

- **Should PREPARE depend on `reports/data_quality.json`?** Yes. It creates the order VALIDATE -> PREPARE: if the gate fails, VALIDATE stops and PREPARE never runs; if the report content changes, PREPARE is re-evaluated.
- **Should PREPARE also depend directly on `data/raw`?** Yes. `prepare.py` actually reads the images; the report only says they were valid (it is the same JSON for different valid files). Without the direct dependency, replacing images with other valid ones would not trigger PREPARE.
- **Which `params.yaml` changes make TRAIN run again?** Only the individual keys it uses: `data.image_size`, `train.batch_size/epochs/learning_rate/dropout/random_state`, `augmentation.enabled/horizontal_flip/rotation_degrees` and `compute.device`. Not `evaluation.*`, `selection.run_id` or `tracking.*`.
- **Why does EVALUATE track `selection.run_id`?** Because it changes which model is scored. If the chosen Run changes, the saved `test_metrics.json` no longer matches and must be recomputed. (`validate_selected_run` also reads `data.image_size`, `tracking.experiment`; the loader reads `evaluation.batch_size`.)
- **Which files are metrics?** `reports/train_summary.json` (TRAIN) and `reports/test_metrics.json` (EVALUATE). Small `reports/` files use `cache: false` so Git stores them; processed images, manifests and `models/model.pt` stay in the DVC cache.
- Code dependencies include indirect imports: EVALUATE and TRAIN list `src/data.py` and `src/reproducibility.py` (imported through the data loader), `src/provenance.py` and `src/model.py`. `uv.lock` + `pyproject.toml` are dependencies of every stage because they define the Python environment. Dependencies point to `data/raw`, never to `data/raw.dvc`.
- Processed data is split into `train/`, `validation/`, `test/`, `manifests/` outputs so TRAIN depends only on `train`, `validation` and the two matching CSVs, while EVALUATE depends on `test` and the manifests.

## Milestone 5 - Provenance and MLflow logging

- `src/provenance.py`: `build_run_manifest()` collects the Git commit SHA and clean/dirty flag (`git rev-parse HEAD`, `git status --porcelain` through `subprocess`; explicit `"unavailable"` if git fails). Changes that are only pipeline outputs (`reports/`, `models/`, `data/processed/`, `dvc.lock`) are listed under `ignored_pipeline_outputs` and do not set `dirty`: DVC deletes the Git-tracked outputs of a stage right before running it, so otherwise every Run would look dirty, `training_data_identity`, OS/platform, Python, torch, torchvision, DVC and MLflow versions (`platform`, `importlib.metadata`), CUDA availability/runtime, device, seed, image size, batch size, epochs, learning rate, dropout, augmentation settings and the MLflow Run ID.
- `src/train.py`, in the required order: one Run per execution in `bean-image-classification`; `write_run_manifest()` called first (before any TRAIN output exists); `training_data_identity` read from the manifest; the 11 required parameters logged; `train.csv` logged as MLflow Dataset input (`name=bean-training`, `context=training`, `targets=label`); `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1` logged with `step=epoch`; `training_seconds` logged; the three artifacts (`training_curves.png`, `train_summary.json`, `run_manifest.json`) logged; model exported on CPU in eval mode with an input example from the **validation** loader, an inferred signature (float32 NCHW input -> logits) and `metadata={class_names, image_size}` as `bean-cnn` (`serialization_format="pickle"`, `code_paths=["src"]`). The test set is never touched by TRAIN.
- `uv run --locked pytest -q` -> `19 passed`.

**Engineering checkpoint 4**

- **Git** records the exact code, configuration (`params.yaml`, `dvc.yaml`) and their history - identified by a commit SHA.
- **DVC** records which data and intermediate files (by content hash in `dvc.lock`/`*.dvc`) and which stage dependencies/parameters produced which outputs, and can rebuild or skip stages accordingly.
- **MLflow** records what happened inside one training Run: parameters, per-epoch metrics, timing, artifacts, the dataset input, and the logged model with its signature - plus the run manifest linking back to the Git SHA and data identity.

## Milestone 6 - Which stages run again?

### 6.1 Predictions (written before running DVC)

Assumption: all outputs are up to date, one change at a time, `selection.run_id` fixed.

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py`; the quality report stays the same | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

**Engineering checkpoint 5**

- *Setting change (`train.learning_rate`)*: only `train` lists `train.learning_rate` under `params:`, so its recorded value in `dvc.lock` changes and TRAIN reruns. `validate` and `prepare` do not list it (they only use `data.image_size` / nothing) and `evaluate` does not list it either: it scores the fixed, already-selected MLflow Run (`selection.run_id` unchanged) and does not read `models/model.pt`, so the saved test results remain valid -> REUSED.
- *Setting change (`data.image_size`)*: `prepare` and `train` and `evaluate` list it. PREPARE resizes the images to the new size, so its outputs change and TRAIN/EVALUATE inputs (`data/processed/...`) change as well. For EVALUATE, DVC will rerun, but the selected Run was trained at the old size, so the saved test results would no longer match and the evaluator would reject the Run -> NEEDS UPDATE (a new Run and a new `selection.run_id` are needed). VALIDATE does not read this setting -> REUSED.
- *File edit (`src/validate_data.py`)*: it is a dependency of `validate` only, so VALIDATE reruns. It regenerates a byte-identical `reports/data_quality.json`; since this is PREPARE's only changed-candidate input and its hash is unchanged (and `data/raw` is unchanged), PREPARE, and therefore TRAIN and EVALUATE, are REUSED.
- *File edit (`README.md`)*: it is not a dependency of any stage, so nothing reruns.

### 6.3 Decisions made before training

- **Primary validation metric:** `val_macro_f1` at the last epoch (the logged model is the last-epoch model).
- **Tie-break rule (Milestone 7/8):** if two Runs differ by less than 0.005 macro-F1, choose the one with the lower final `val_loss`; if still tied, the higher `val_accuracy`; then the shorter `training_seconds`.
- **Expectation for a lower learning rate (0.0003 vs 0.001):** with only 4 epochs, Adam with a 3x smaller step should converge more slowly, so Run B should have a *higher* final validation loss and a *lower* (or at best equal) validation accuracy / macro-F1 than Run A. Training time should be about the same, since only the step size changes.

### 6.3 Baseline - Run A

`uv run --locked dvc repro train` with the default settings (MLflow server on `http://127.0.0.1:5000`, experiment `bean-image-classification`).

**Run A MLflow Run ID: `15ee55bab76c47a2821a86e42b1fb55f`** (status FINISHED).

Checked in MLflow for Run A: the 11 required parameters (image_size 96, batch_size 32, epochs 4, learning_rate 0.001, dropout 0.3, random_state 42, augmentation_enabled False, horizontal_flip 0.5, rotation_degrees 10.0, device cpu, training_data_identity); `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1` with 4 points each (steps 1-4) and `training_seconds` = 17.9 s; the Dataset input `bean-training` (context `training`); the artifacts `run_manifest.json`, `train_summary.json`, `training_curves.png`; the model `bean-cnn` with metadata `{class_names: [angular_leaf_spot, bean_rust, healthy], image_size: 96}` and signature `float32 (-1,3,96,96) -> float32 (-1,3)`.

Epochs of Run A: macro-F1 0.5562 / 0.6119 / 0.6445 / **0.6254 (final)**; final val_acc 0.6391, final val_loss 0.8324.

Manifest: commit `758d90e`, `dirty: false`, `changed_files: []` (only the pipeline outputs `dvc.lock` and `reports/data_summary.json` were new, listed under `ignored_pipeline_outputs`).

Note: two earlier passes of this milestone showed a wrong `dirty: true`: (1) DVC deletes the Git-tracked stage outputs before running a stage, so Git always saw changes; (2) my `git status --porcelain` parser stripped the leading space of the first line and truncated the first path. I fixed both in `provenance.py` (see Milestone 5, checked on a toy repository with a deleted output, a modified file and an untracked file), discarded those passes and replayed the milestone from a fresh MLflow database; the metrics were identical each time (training is deterministic with the fixed seed), which is itself a small repeatability check.

Environment note: the lab was executed in a Linux sandbox that could not reach the PyTorch CPU wheel index, so PyTorch/torchvision 2.11.0/0.26.0 came from PyPI (CUDA build running on CPU, `CUDA available: False`) instead of the `+cpu` wheels in `uv.lock`. The manifest records the versions actually used. Re-run once with `uv sync --locked` on the lab machine to confirm the numbers.

**Check 1 - rerun without any change**

```
'data/raw.dvc' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Data and pipelines are up to date.
```

**Check 2 - comment added to `src/validate_data.py` (report unchanged)**

```
'data/raw.dvc' didn't change, skipping
Running stage 'validate':
> uv run --locked python -m src.validate_data
Updating lock file 'dvc.lock'
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
```

`reports/data_quality.json` stayed byte-identical (`cmp`), so PREPARE and TRAIN were skipped. The comment stays in place.

### 6.4 Run B - learning rate 0.0003

`params.yaml` changed to `train.learning_rate: 0.0003` and committed alone, then `uv run --locked dvc repro train`.

**Run B MLflow Run ID: `3d40acc0260b4c7aa5f24cf5b2b2dc89`.** Final epoch: val_acc 0.6165, val_macro_f1 0.6091, val_loss 0.8927 (epoch-1 macro-F1 was only 0.4723).

Observed DVC behaviour (all checks use `dvc repro train`, so EVALUATE was never started):

| Check | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| 1. Rerun, nothing changed | REUSED | REUSED | REUSED | NOT RUN |
| 2. Comment in `src/validate_data.py` | RERUN | REUSED | REUSED | NOT RUN |
| 3. `train.learning_rate` 0.001 -> 0.0003 | REUSED | REUSED | RERUN | NOT RUN |

Terminal lines:

```
# Check 1
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Data and pipelines are up to date.

# Check 2
Running stage 'validate':
> uv run --locked python -m src.validate_data
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping

# Check 3
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Running stage 'train':
> uv run --locked python -m src.train
```

**Comparison with the predictions (6.1).** Setting change: VALIDATE REUSED, PREPARE REUSED, TRAIN RERUN, as predicted. File edit in `validate_data.py`: VALIDATE RERUN, PREPARE REUSED (the report hash was unchanged), TRAIN REUSED, as predicted. README row and `image_size` row were not executed (predictions only). For EVALUATE the predictions said REUSED / NEEDS UPDATE, whereas the table says NOT RUN: that is only because `dvc repro train` never touches EVALUATE; with a full `dvc repro`, DVC would also have judged it REUSED for check 3 (no dependency changed). No difference in substance.

**Learning-rate expectation vs result.** Predicted: slower convergence, higher final val loss, lower or equal accuracy/F1. Observed: B has val_loss 0.8927 vs 0.8324, accuracy 0.6165 vs 0.6391 and macro-F1 0.6091 vs 0.6254, so the prediction is supported (the small step size is simply too small for only 4 epochs); the gap is small though.

## Milestone 7 - Three controlled runs

### 7.1 Run C setup

`train.learning_rate` was put back to 0.001 (Run A's value). **Exactly one change versus A: `train.batch_size` 32 -> 16.**

Expectation (written before training): with batch size 16 there are twice as many optimizer updates per epoch (about 65 instead of 33), so within a fixed budget of 4 epochs the model should converge faster. I expect a lower final validation loss and a higher macro-F1 than Run A, at the cost of a slightly longer training time (smaller batches are less efficient on CPU) and noisier curves. I do not expect a large gap with a single seed and 133 validation images.

### 7.2 Validation comparison (last-epoch scores, same epoch as the logged model)

| Run | Controlled change | Val. Accuracy | Val. Macro-F1 | Val. loss | Training time | MLflow Run ID |
|---|---|---|---|---|---|---|
| A | Baseline (lr 0.001, batch 32) | 0.6391 | 0.6254 | 0.8324 | 17.9 s | `15ee55bab76c47a2821a86e42b1fb55f` |
| B | `learning_rate` 0.0003 | 0.6165 | 0.6091 | 0.8927 | 16.2 s | `3d40acc0260b4c7aa5f24cf5b2b2dc89` |
| C | `batch_size` 16 | 0.6391 | 0.6349 | 0.8134 | 17.6 s | `aa36785cac4b4bc795cf9d98c205f9a6` |

All three Runs are FINISHED, trained on the same data identity, with seed 42 and 4 epochs; the parameters in MLflow confirm that only the intended setting differs.

- **B vs A:** lower accuracy (-0.0226), lower macro-F1 (-0.0163) and higher loss (+0.0603). The prediction (slower convergence at lr 0.0003 in 4 epochs) is supported; the loss curve of B is still falling steeply at epoch 4, so B is under-trained rather than "worse" in general.
- **C vs A:** same accuracy (0.6391), macro-F1 +0.0095 and val loss -0.0190. The prediction (faster convergence with more updates per epoch) is directionally supported, and training time was essentially unchanged (17.6 s vs 17.9 s, so my "slightly longer" guess was not confirmed). The difference is tiny: with 133 validation images, one image is 0.75 points of accuracy, and with a single seed this gap could be noise. Equal accuracy with different macro-F1 shows the two models make errors in a different class mix.
- Absolute scores are modest (about 0.63 F1) because the CNN is small and only trained for 4 epochs; the aim of this lab is the traceability, not the best model.

**Engineering checkpoint 6**

- *Primary metric:* `val_macro_f1`. The three classes are almost balanced but the model errs differently across them (A and C have the same accuracy and different F1); macro-F1 averages the per-class F1 equally, so it exposes a weak class (e.g. a disease confused with another) that accuracy can hide, and it does not favour a majority class. Validation loss is used as the tie-breaker because it is more sensitive than a discrete score on 133 images.
- *Test set only after choosing:* if the test set were used to compare candidates, selection would be tuned to it, the final score would be optimistically biased, and nothing unbiased would remain to estimate how the chosen model behaves on new data. The test split is therefore touched exactly once, by EVALUATE, for the single frozen Run.

```
git add dvc.lock reports/ LAB_REPORT.md  # commit "Compare three training runs"
```
