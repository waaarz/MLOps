from pathlib import Path
import shutil

from PIL import Image
import pytest

from src.validate_data import validate_dataset


def tiny_dataset(root, monkeypatch):
    import src.validate_data as vd
    monkeypatch.setattr(vd, 'EXPECTED_SPLITS', {'train': 3, 'validation': 3, 'test': 3})
    for split in vd.EXPECTED_SPLITS:
        for cls in vd.EXPECTED_CLASSES:
            folder = root / split / cls
            folder.mkdir(parents=True)
            Image.new('RGB', (12, 12), (30, 70, 90)).save(folder / 'image.jpg')


@pytest.mark.parametrize('fault', ['missing_split', 'extra_class', 'empty_class', 'wrong_count'])
def test_dataset_contract_rejects_structural_faults(tmp_path, monkeypatch, fault):
    tiny_dataset(tmp_path, monkeypatch)
    if fault == 'missing_split':
        shutil.rmtree(tmp_path / 'validation')
    elif fault == 'extra_class':
        (tmp_path / 'train' / 'unexpected').mkdir()
    elif fault == 'empty_class':
        (tmp_path / 'test/healthy/image.jpg').unlink()
    else:
        Image.new('RGB', (12, 12)).save(tmp_path / 'train/healthy/extra.jpg')
    with pytest.raises((RuntimeError, FileNotFoundError)) as raised:
        validate_dataset(tmp_path)
    assert not isinstance(raised.value, NotImplementedError), 'Implement the validator before testing rejection rules'


def test_quality_report_is_deterministic(tmp_path, monkeypatch):
    tiny_dataset(tmp_path, monkeypatch)
    assert validate_dataset(tmp_path) == validate_dataset(tmp_path)
