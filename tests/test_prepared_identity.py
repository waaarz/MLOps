import hashlib
from pathlib import Path

from PIL import Image
import pandas as pd
import yaml

from src.prepare import main as prepare
from src.provenance import training_data_identity


def test_prepared_bytes_are_part_of_data_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('params.yaml').write_text(yaml.safe_dump({'data': {'image_size': 5}}))
    for split in ('train', 'validation', 'test'):
        folder = Path('data/raw') / split / 'healthy'
        folder.mkdir(parents=True)
        image = Image.new('RGB', (16, 16))
        image.putdata([((255, 255, 255) if (x+y) % 2 else (0, 0, 0))
                       for y in range(16) for x in range(16)])
        image.save(folder / 'leaf.png')
    prepare()
    first = pd.read_csv('data/processed/manifests/train.csv')
    original_identity = training_data_identity()
    actual = hashlib.sha256(Path(first.iloc[0].relative_path).read_bytes()).hexdigest()
    assert first.iloc[0].processed_sha256 == actual
    # A preprocessing implementation changes while raw bytes, paths and image size stay fixed.
    original_resize = Image.Image.resize
    monkeypatch.setattr(Image.Image, 'resize', lambda self, size, *args, **kwargs:
                        original_resize(self, size, resample=Image.Resampling.NEAREST))
    prepare()
    changed = pd.read_csv('data/processed/manifests/train.csv')
    assert first.iloc[0].source_sha256 == changed.iloc[0].source_sha256
    assert original_identity != training_data_identity()
