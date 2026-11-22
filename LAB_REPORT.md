# Laboratory Report (short version)

## Milestone 0 - Audit of the starter

| File / setting | Problem | Fix |
|---|---|---|
| `src/reproducibility.py` | Generator and worker seeds were not really seeded | Seed the generator and derive worker seeds from `torch.initial_seed()` |
| `src/validate_data.py` | Data-quality gate not implemented | Check splits, classes, counts, full image decoding |
| `src/provenance.py` | No run manifest (Git, data, versions, settings) | Implement the manifest |
| `src/train.py` | Nothing was logged to MLflow | Complete the TODOs (params, metrics, artifacts, model) |
| `dvc.yaml` | No pipeline | Write the 4 stages |
| `params.yaml` - `selection.run_id` | Empty, so no model is selected | Set it after comparing runs |
| Git | No commits at the start | Commit after each milestone |

**Checkpoint 1 - Why is a working `model.pt` not enough?**
It does not say which code, data, package versions, settings and seed produced it. Without them nobody can rerun the experiment and check the result.

## Milestone 1 - Repeatable training

`pyproject.toml` lists the dependencies, `uv.lock` fixes the exact versions, `.venv/` is the local environment. I seeded Python, NumPy, PyTorch (CPU and CUDA), the DataLoader generator and the workers. The reproducibility tests pass.

**Checkpoint 2 - Does `random_state=42` give identical results everywhere?**
No. Hardware, library versions and some non-deterministic GPU operations can change the results. A seed only gives repeatability on the same setup, so the setup must be recorded too.

## Milestone 2 - Track the images

I extracted the Beans archives (1034 train / 133 validation / 128 test), then ran `dvc init`, `dvc add data/raw` and set the DVC remote.

**Checkpoint 3**
- The Git remote stores code, settings and the small `.dvc` pointer files.
- The DVC remote stores the real data files.
- `git push` does not send the raw images; only `dvc push` does.

## Milestone 3 - Data-quality gate

`validate_dataset()` stops with an error if a split is missing, the classes are wrong, a count is wrong, a class is empty, or an image cannot be fully decoded. The report has no timestamp, so it is identical on every run. Result: `PASS`, 1295 images.

## Milestone 4 - DVC pipeline

Stages: `validate -> prepare -> train`, and `prepare -> evaluate` (evaluate does not depend on train).

- PREPARE depends on the quality report (so it runs after the gate) and directly on `data/raw` (it reads the images).
- TRAIN reruns only when a setting it uses changes (image size, training settings, augmentation, device).
- EVALUATE tracks `selection.run_id` because it decides which model is tested.
- Metrics files: `reports/train_summary.json` and `reports/test_metrics.json`.

## Milestone 5 - Provenance and MLflow

Each training Run writes a manifest first (Git commit and clean/dirty state, data identity, library versions, settings, Run ID). It then logs the parameters, the training dataset, per-epoch metrics, training time, three artifacts and the `bean-cnn` model with its signature and metadata. The test set is never used in training. All 19 tests pass.

**Checkpoint 4**
- Git records the code and settings.
- DVC records the data and which stage produced which file.
- MLflow records what happened in each training run (parameters, metrics, files, model).

## Milestone 6 - Which stages rerun?

Predictions made before running DVC:

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `validate_data.py` (same report) | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

**Checkpoint 5 (two rows)**
- *Learning rate:* only TRAIN lists this setting, so only TRAIN reruns. EVALUATE tests a fixed, already selected model, so its results stay valid.
- *Editing `validate_data.py`:* only VALIDATE depends on that file, so it reruns. It produces the same report, so PREPARE and what follows are reused.

Decisions before training: main metric = `val_macro_f1` (last epoch); tie-break = lower `val_loss`. I expected a lower learning rate to give a slower, slightly worse model after 4 epochs.

**Run A (baseline), ID `15ee55bab76c47a2821a86e42b1fb55f`.** Checks in MLflow were all fine.

Observed DVC behaviour:

| Check | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| Rerun, nothing changed | REUSED | REUSED | REUSED | NOT RUN |
| Comment in `validate_data.py` | RERUN | REUSED | REUSED | NOT RUN |
| Learning rate 0.001 -> 0.0003 (Run B) | REUSED | REUSED | RERUN | NOT RUN |

The results match my predictions. (EVALUATE is "NOT RUN" only because `dvc repro train` does not touch it.)

## Milestone 7 - Three controlled runs

Run C changes only `batch_size` (32 -> 16). I expected faster learning because of more updates per epoch.

| Run | Change | Val. Acc. | Val. Macro-F1 | Time | Run ID |
|---|---|---|---|---|---|
| A | Baseline | 0.6391 | 0.6254 | 17.9 s | `15ee55bab76c47a2821a86e42b1fb55f` |
| B | Learning rate 0.0003 | 0.6165 | 0.6091 | 16.2 s | `3d40acc0260b4c7aa5f24cf5b2b2dc89` |
| C | Batch size 16 | 0.6391 | 0.6349 | 17.6 s | `aa36785cac4b4bc795cf9d98c205f9a6` |

- **B vs A:** worse, as predicted (the model is under-trained after 4 epochs).
- **C vs A:** slightly better F1 and loss, as predicted. The gap is small (one seed, 133 images), so it may be noise.

**Checkpoint 6**
- *Metric:* macro-F1 treats the three classes equally and shows a weak class that accuracy can hide (A and C have the same accuracy but different F1).
- *Test set last:* using it to choose a model would make the final score too optimistic. It is used once, on the frozen choice.

## Milestone 8 - Selection and final test

**Selected: Run C, `aa36785cac4b4bc795cf9d98c205f9a6`** (best validation macro-F1 0.6349 and lowest validation loss). Limitation: its lead over A is small, so it is "best on this evidence", not proven better.

**Final test result (evaluated once):**

| Run ID | Test accuracy | Test macro-F1 |
|---|---|---|
| `aa36785cac4b4bc795cf9d98c205f9a6` | 0.7031 | 0.6973 |

**For Phase 2:** keep `mlflow.db`, `mlartifacts/` and the DVC storage folder. The model is `runs:/aa36785cac4b4bc795cf9d98c205f9a6/bean-cnn`.

*Note:* the results were produced with PyTorch from PyPI instead of the CPU wheels in `uv.lock`. Rerun once with `uv sync --locked` to confirm the numbers.
