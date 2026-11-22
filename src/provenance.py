from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


UNAVAILABLE = "unavailable"


def _git(*args: str) -> str | None:
    """Run a git command; return stripped stdout, or None if git/the repo is unusable."""
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    # Strip only trailing newlines: the first column of `git status --porcelain` can be a
    # space (" M file") and must be kept so that paths are parsed correctly.
    return result.stdout.rstrip("\r\n")


# Files that the DVC pipeline itself (re)writes. DVC deletes the stage outputs that Git
# tracks (reports/*.json, dvc.lock ...) right before running a stage, so Git always sees
# them as modified/deleted during `dvc repro`. They are results, not code or settings,
# so they must not make the Run look "dirty".
PIPELINE_OUTPUT_PREFIXES = ("reports/", "models/", "data/processed/")
PIPELINE_OUTPUT_FILES = ("dvc.lock",)


def _is_pipeline_output(path: str) -> bool:
    return path in PIPELINE_OUTPUT_FILES or path.startswith(PIPELINE_OUTPUT_PREFIXES)


def _git_state() -> dict:
    """Commit SHA and clean/dirty state of the working tree (explicit values if git fails).

    `dirty` is True when code, configuration or data pointers differ from the commit.
    Changes limited to pipeline outputs are listed separately in `ignored_pipeline_outputs`.
    """
    sha = _git("rev-parse", "HEAD")
    sha = sha.strip() if sha is not None else None
    status = _git("status", "--porcelain")
    if sha is None or status is None:
        return {"commit_sha": UNAVAILABLE, "working_tree": UNAVAILABLE, "dirty": None}
    changed, outputs = [], []
    for line in status.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:  # renames: "old -> new"
            path = path.split(" -> ")[-1]
        (outputs if _is_pipeline_output(path) else changed).append(path)
    dirty = bool(changed)
    return {
        "commit_sha": sha,
        "working_tree": "dirty" if dirty else "clean",
        "dirty": dirty,
        "changed_files": sorted(changed),
        "ignored_pipeline_outputs": sorted(outputs),
    }


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return UNAVAILABLE


def _environment() -> dict:
    import torch  # local import keeps this module light for the evaluator

    return {
        "platform": platform.platform(),
        "os": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "dvc": _package_version("dvc"),
        "mlflow": _package_version("mlflow"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda if torch.cuda.is_available() else None,
    }


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata for one training Run.

    Records the code (Git SHA + dirty flag), the data (identity of the prepared
    training + validation manifests), the environment (OS, Python, library
    versions, CUDA), the settings (device, seed, image size, hyper-parameters,
    augmentation) and the MLflow Run ID. Unavailable values are written
    explicitly instead of failing, so the manifest is always produced.
    """
    train = params["train"]
    aug = params["augmentation"]
    return {
        "git": _git_state(),
        "data": {"training_data_identity": training_data_identity()},
        "environment": _environment(),
        "execution": {
            "mlflow_run_id": mlflow_run_id,
            "device": device,
            "random_state": int(train["random_state"]),
        },
        "configuration": {
            "image_size": int(params["data"]["image_size"]),
            "batch_size": int(train["batch_size"]),
            "epochs": int(train["epochs"]),
            "learning_rate": float(train["learning_rate"]),
            "dropout": float(train["dropout"]),
            "augmentation": {
                "enabled": bool(aug["enabled"]),
                "horizontal_flip": float(aug["horizontal_flip"]),
                "rotation_degrees": float(aug["rotation_degrees"]),
            },
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
