from pathlib import Path

from PIL import Image, UnidentifiedImageError
import pytest

from src.validate_data import validate_dataset


def create_tiny_dataset(root: Path):
    for split in ["train", "validation", "test"]:
        for cls in ["angular_leaf_spot", "bean_rust", "healthy"]:
            d = root / split / cls
            d.mkdir(parents=True)
            Image.new("RGB", (8, 8), color=(20, 40, 60)).save(d / "x.jpg")


def patch_tiny_counts(monkeypatch):
    import src.validate_data as vd
    monkeypatch.setattr(vd, "EXPECTED_SPLITS", {"train": 3, "validation": 3, "test": 3})


def test_validator_rejects_corrupt_image(tmp_path, monkeypatch):
    create_tiny_dataset(tmp_path)
    patch_tiny_counts(monkeypatch)
    (tmp_path / "test" / "healthy" / "x.jpg").write_bytes(b"not-an-image")
    with pytest.raises((OSError, UnidentifiedImageError)):
        validate_dataset(tmp_path)


def test_validator_accepts_valid_dataset(tmp_path, monkeypatch):
    create_tiny_dataset(tmp_path)
    patch_tiny_counts(monkeypatch)
    report = validate_dataset(tmp_path)
    assert report["status"] == "PASS"
    assert report["total_images"] == 9
    assert set(report["splits"]) == {"train", "validation", "test"}
