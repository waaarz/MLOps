"""Exercise the evaluator's rejection boundary without touching the test set."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluate import validate_selected_run
from src.provenance import training_data_identity


def configured_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifests = Path('data/processed/manifests')
    manifests.mkdir(parents=True)
    for split in ('train', 'validation'):
        (manifests / f'{split}.csv').write_text('relative_path,label,processed_sha256\na.jpg,healthy,abc\n')
    params = {'data': {'image_size': 96}, 'tracking': {'experiment': 'course'}}
    run = SimpleNamespace(info=SimpleNamespace(status='FINISHED', experiment_id='1'),
                          data=SimpleNamespace(params={'image_size': '96', 'training_data_identity': training_data_identity()}))
    client = SimpleNamespace(get_run=lambda _: run,
                             get_experiment_by_name=lambda _: SimpleNamespace(experiment_id='1'))
    return params, run, client


def test_matching_selected_run_is_accepted(tmp_path, monkeypatch):
    params, run, client = configured_run(tmp_path, monkeypatch)
    validate_selected_run(client, 'chosen-run', params)


@pytest.mark.parametrize('mismatch', ['status', 'experiment', 'size', 'identity', 'missing_identity'])
def test_invalid_selected_run_is_rejected(tmp_path, monkeypatch, mismatch):
    params, run, client = configured_run(tmp_path, monkeypatch)
    if mismatch == 'status':
        run.info.status = 'FAILED'
    elif mismatch == 'experiment':
        run.info.experiment_id = 'other'
    elif mismatch == 'size':
        run.data.params['image_size'] = '128'
    elif mismatch == 'identity':
        Path('data/processed/manifests/train.csv').write_text('different prepared data\n')
    else:
        del run.data.params['training_data_identity']
    with pytest.raises(RuntimeError):
        validate_selected_run(client, 'chosen-run', params)
