import json
import time

import pytest
import torch

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
from aether.live.training import TrainingJob, retention_eligible
from aether.evidence.receipts import verify_receipt


def model():
    return TissueModel(ModelConfig(neuron_count=16, scalar_edges=48, region_count=4,
                                    region_message_edges=8, state_dimension=6,
                                    mode_count=2, complex_dimensions=2), 'wave', 11)


def wait_for_job(job, timeout=30):
    deadline = time.monotonic() + timeout
    while job.snapshot()['status'] in ('running', 'evaluating'):
        if time.monotonic() >= deadline:
            job.cancel()
            pytest.fail('training job did not complete within bounded test timeout')
        time.sleep(.02)
    return job.snapshot()


def test_candidate_training_isolated_paired_and_saved(tmp_path):
    active = model()
    before = active.state_digest()
    job = TrainingJob()
    job.start(active, updates=3, output_root=tmp_path, delay=3)
    result = wait_for_job(job)
    assert result['status'] == 'ready', result
    assert result['progress'] == 3
    assert len(result['curve']) == 3
    assert active.state_digest() == before == result['source_digest']
    assert job.candidate_model.state_digest() != before
    output = tmp_path / result['candidate']['id']
    assert verify_receipt(output / 'receipt.json')['valid']
    record = json.loads((output / 'training.json').read_text())
    assert record['before']['query']['episode_seeds'] == record['after']['query']['episode_seeds']
    assert record['before']['retention']['episode_seeds'] == record['after']['retention']['episode_seeds']
    assert not set(record['support_seeds']) & set(record['after']['query']['episode_seeds'])
    assert not set(record['support_seeds']) & set(record['after']['retention']['episode_seeds'])
    assert record['executed_support_steps'] > 0
    assert record['promoted'] is False
    saved = torch.load(output / 'candidate.pt', weights_only=True)
    assert saved['schema'] == 'aether.live.candidate.v1'
    assert saved['trained_digest'] == job.candidate_model.state_digest()


@pytest.mark.parametrize('updates', [0, 101, -1, True, 1.5])
def test_bad_training_request_does_not_change_idle_job(tmp_path, updates):
    job = TrainingJob()
    with pytest.raises(ValueError):
        job.start(model(), updates, tmp_path)
    assert job.snapshot()['status'] == 'idle'
    assert list(tmp_path.iterdir()) == []


def test_retention_gate_is_not_scientific_promotion():
    assert retention_eligible(.5, .5)
    assert retention_eligible(.5, .48)
    assert not retention_eligible(.5, .4)
    for score in (float('nan'), float('inf'), -.1, 1.1):
        with pytest.raises(ValueError):
            retention_eligible(score, .5)


def test_cancel_job_and_reject_concurrent_start(tmp_path):
    active = model()
    before = active.state_digest()
    job = TrainingJob()
    job.start(active, 100, tmp_path)
    with pytest.raises(RuntimeError):
        job.start(active, 1, tmp_path)
    job.cancel(wait=True)
    result = wait_for_job(job)
    assert result['status'] == 'cancelled'
    assert job.candidate_model is None
    assert active.state_digest() == before
    assert not result['can_apply']
