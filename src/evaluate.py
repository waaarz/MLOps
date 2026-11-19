from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from src.data import build_test_loader
from src.provenance import training_data_identity


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("compute.device=cuda but CUDA is not available")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError("compute.device must be auto, cpu or cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_selected_run(client: MlflowClient, run_id: str, params: dict) -> None:
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise RuntimeError("Selected Run must have status FINISHED before final evaluation")
    expected_experiment = client.get_experiment_by_name(params["tracking"]["experiment"])
    if expected_experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {params['tracking']['experiment']}")
    if run.info.experiment_id != expected_experiment.experiment_id:
        raise RuntimeError("Selected Run does not belong to the configured course experiment")

    logged_size = run.data.params.get("image_size")
    current_size = str(params["data"]["image_size"])
    if logged_size is None:
        raise RuntimeError("Selected Run is missing the logged image_size parameter")
    if str(logged_size) != current_size:
        raise RuntimeError(
            f"Selected Run image_size={logged_size} does not match current data.image_size={current_size}. "
            "Restore the matching configuration before final test evaluation."
        )

    logged_identity = run.data.params.get("training_data_identity")
    current_identity = training_data_identity()
    if logged_identity is None:
        raise RuntimeError("Selected Run is missing the logged training_data_identity parameter")
    if current_identity == "unavailable":
        raise RuntimeError(
            "Current training-data identity is unavailable; restore the prepared "
            "train/validation manifests before evaluation"
        )
    if str(logged_identity) != str(current_identity):
        raise RuntimeError(
            "Selected Run was trained from a different prepared training/validation data state. "
            "Restore the matching prepared data or select a Run from the current data state."
        )


def main() -> None:
    # MLflow prints Unicode symbols; DVC may capture stdout using a legacy Windows encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    run_id = str(params["selection"]["run_id"]).strip()
    if not run_id:
        raise RuntimeError(
            "selection.run_id is empty. Select a validation Run in MLflow before final test evaluation."
        )

    mlflow.set_tracking_uri(params["tracking"]["uri"])
    client = MlflowClient()
    validate_selected_run(client, run_id, params)
    device = resolve_device(str(params["compute"]["device"]))

    # The final model comes from the human-selected MLflow Run.
    # models/model.pt is a local DVC-managed checkpoint, not the selection authority.
    model_uri = f"runs:/{run_id}/bean-cnn"
    print(f"Loading selected model: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model = model.to(device).eval()

    test_loader, class_names = build_test_loader(
        Path("data/processed"), int(params["evaluation"]["batch_size"])
    )

    model_info = mlflow.models.get_model_info(model_uri)
    metadata = model_info.metadata or {}
    if metadata.get("class_names") != list(class_names):
        raise RuntimeError("Selected model class_names do not match the test class ordering")
    if metadata.get("image_size") != int(params["data"]["image_size"]):
        raise RuntimeError("Selected model metadata image_size does not match preprocessing")

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            y_true.extend(targets.tolist())
            y_pred.extend(predictions.tolist())

    metrics = {
        "selected_run_id": run_id,
        "training_data_identity": training_data_identity(),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/test_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    labels = [name.replace("_", "\n") for name in class_names]
    with plt.rc_context({"font.size": 10, "axes.titleweight": "bold"}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        for ax, normalization, title, formatting in (
            (axes[0], None, "Image counts", "d"),
            (axes[1], "true", "Recall by true class", ".2f"),
        ):
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, labels=list(range(len(class_names))), display_labels=labels,
                normalize=normalization, values_format=formatting,
                cmap="Blues", colorbar=False, ax=ax,
            )
            ax.set_title(title)
        fig.suptitle("Final test | selected Run " + run_id[:8], color="#102A43")
        fig.savefig("reports/test_confusion_matrix.png", dpi=180)
        plt.close(fig)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
