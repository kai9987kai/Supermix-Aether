import json
import time

import pytest
import torch

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
from aether.live.training import TrainingJob, retention_eligible
from aether.evidence.receipts import verify_receipt
from aether.environments.stress import TASK_FAMILIES, stress_expert_batch


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


@pytest.mark.parametrize('task_family,expected_steps', [
    ('straight', 120), ('distraction', 120), ('waypoint', 168),
])
def test_candidate_training_isolated_paired_and_saved(tmp_path, task_family, expected_steps):
    active = model()
    before = active.state_digest()
    support = stress_expert_batch([3_000_110_000 + i for i in range(8)],
                                 task_family=task_family, delay=3, horizon=9)
    with torch.no_grad():
        logits = active(torch.from_numpy(support['observations']), disturbance_seed=11000)
        mask = torch.from_numpy(support['mask'])
        expected_first_loss = float(torch.nn.functional.cross_entropy(
            logits[mask], torch.from_numpy(support['actions'])[mask]))
    job = TrainingJob()
    job.start(active, updates=3, output_root=tmp_path, delay=3, task_family=task_family)
    result = wait_for_job(job)
    assert result['status'] == 'ready', result
    assert result['progress'] == 3
    assert len(result['curve']) == 3
    assert result['task_family'] == task_family
    assert result['curve'][0]['loss'] == pytest.approx(expected_first_loss)
    assert active.state_digest() == before == result['source_digest']
    assert job.candidate_model.state_digest() != before
    output = tmp_path / result['candidate']['id']
    assert verify_receipt(output / 'receipt.json')['valid']
    record = json.loads((output / 'training.json').read_text())
    assert record['before']['query']['episode_seeds'] == record['after']['query']['episode_seeds']
    assert record['before']['retention']['episode_seeds'] == record['after']['retention']['episode_seeds']
    assert not set(record['support_seeds']) & set(record['after']['query']['episode_seeds'])
    assert not set(record['support_seeds']) & set(record['after']['retention']['episode_seeds'])
    assert record['task_family'] == task_family
    assert record['executed_support_steps'] == expected_steps
    assert set(record['before']['task_matrix']) == set(TASK_FAMILIES)
    assert record['before']['query'] == record['before']['task_matrix'][task_family]['query']
    assert record['after']['retention'] == record['after']['task_matrix'][task_family]['retention']
    for family in TASK_FAMILIES:
        for condition in ('query', 'retention'):
            before_evaluation = record['before']['task_matrix'][family][condition]
            after_evaluation = record['after']['task_matrix'][family][condition]
            assert before_evaluation['episode_seeds'] == after_evaluation['episode_seeds']
            assert before_evaluation['episode_seeds'] == record['before'][condition]['episode_seeds']
            assert len(after_evaluation['episodes']) == 12
        gate = result['comparison']['task_matrix'][family]
        assert gate['retention_passed'] == retention_eligible(
            gate['before']['retention_success'], gate['after']['retention_success'])
    assert result['can_apply'] == all(
        gate['retention_passed'] for gate in result['comparison']['task_matrix'].values())
    assert record['promoted'] is False
    saved = torch.load(output / 'candidate.pt', weights_only=True)
    assert saved['schema'] == 'aether.live.candidate.v1'
    assert saved['trained_digest'] == job.candidate_model.state_digest()
    assert saved['task_family'] == task_family


@pytest.mark.parametrize('updates', [0, 101, -1, True, 1.5])
def test_bad_training_request_does_not_change_idle_job(tmp_path, updates):
    job = TrainingJob()
    with pytest.raises(ValueError):
        job.start(model(), updates, tmp_path)
    assert job.snapshot()['status'] == 'idle'
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('task_family', ['unknown', None, True, [], {}])
def test_bad_task_family_does_not_start_or_write(tmp_path, task_family):
    job = TrainingJob()
    with pytest.raises(ValueError, match='task_family'):
        job.start(model(), 1, tmp_path, task_family=task_family)
    assert job.snapshot()['status'] == 'idle'
    assert list(tmp_path.iterdir()) == []


def test_nonactive_task_retention_regression_blocks_candidate(tmp_path, monkeypatch):
    active = model()
    source_digest = active.state_digest()

    def fixed_outcome(model, seeds, *, delay, horizon, task_family, **kwargs):
        # Controlled outcomes isolate the gate: only the unselected waypoint
        # task regresses after training, while the selected straight task holds.
        success = not (model.state_digest() != source_digest and
                       task_family == 'waypoint' and delay == 5)
        episodes = [{'seed': seed, 'success': success, 'return': 1.0 if success else -0.09,
                     'actions': [6] * horizon, 'final_position': [0, 0, 0]} for seed in seeds]
        return {'success_rate': float(success), 'mean_return': episodes[0]['return'],
                'executed_steps': len(seeds) * horizon, 'episode_seeds': list(seeds),
                'episodes': episodes}

    monkeypatch.setattr('aether.live.training.evaluate', fixed_outcome)
    job = TrainingJob()
    job.start(active, 1, tmp_path, task_family='straight')
    result = wait_for_job(job)
    assert result['status'] == 'ready', result
    assert result['comparison']['task_matrix']['straight']['retention_passed']
    assert not result['comparison']['task_matrix']['waypoint']['retention_passed']
    assert not result['can_apply']
    assert not job.can_apply


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
