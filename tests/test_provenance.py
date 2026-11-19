from pathlib import Path

from src.provenance import build_run_manifest, training_data_identity


def test_run_manifest_contains_required_provenance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifests = Path("data/processed/manifests")
    manifests.mkdir(parents=True)
    (manifests / "train.csv").write_text(
        "path,label\ntrain/a.jpg,healthy\n", encoding="utf-8"
    )
    (manifests / "validation.csv").write_text(
        "path,label\nvalidation/b.jpg,bean_rust\n", encoding="utf-8"
    )

    params = {
        "data": {"image_size": 96},
        "train": {
            "batch_size": 32,
            "epochs": 4,
            "learning_rate": 0.001,
            "dropout": 0.3,
            "random_state": 42,
        },
        "augmentation": {
            "enabled": False,
            "horizontal_flip": 0.5,
            "rotation_degrees": 10,
        },
    }

    expected_identity = training_data_identity()
    assert expected_identity != "unavailable"

    manifest = build_run_manifest(params, "cpu", "run-123")
    assert manifest["data"]["training_data_identity"] == expected_identity
    assert manifest["execution"]["mlflow_run_id"] == "run-123"
    assert manifest["execution"]["random_state"] == 42
    assert "python" in manifest["environment"]
    assert "torch" in manifest["environment"]
    assert manifest["configuration"]["image_size"] == 96
