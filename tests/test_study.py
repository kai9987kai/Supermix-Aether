import json
import zipfile

import pytest
import torch

from aether.experiments.study import StudyConfig, run_study, replay_checkpoint


def small_config():
    return StudyConfig(seeds=[3], protocols=['scalar', 'wave'], updates=2,
                       batch_size=2, evaluation_episodes=3,
                       model={'neuron_count': 16, 'scalar_edges': 32,
                              'region_count': 4, 'region_message_edges': 8,
                              'state_dimension': 4, 'mode_count': 2,
                              'complex_dimensions': 2})


def test_study_trains_replays_and_preserves_evidence(tmp_path):
    config = small_config()
    result = run_study(config, tmp_path / 'study')
    assert result['status'] == 'exploratory'
    assert len(result['runs']) == 2
    for run in result['runs']:
        assert run['initial_digest'] != run['trained_digest']
        assert run['query_immutable']
        assert run['executed_support_steps'] > 0
        assert run['evaluation']['executed_steps'] > 0
        replay = replay_checkpoint(tmp_path / 'study' / run['checkpoint'])
        assert replay['valid']
        assert replay['evaluation'] == run['evaluation']
        assert replay['retention'] == run['retention']
        assert replay['interventions'] == run['interventions']
        assert all(s >= 10_000_000 for s in run['evaluation']['episode_seeds'])
    stored = json.loads((tmp_path / 'study' / 'results.json').read_text())
    assert not stored['promotion']['promoted']
    with pytest.raises(FileExistsError):
        run_study(config, tmp_path / 'study')

    checkpoint = tmp_path / 'study' / result['runs'][-1]['checkpoint']
    saved = torch.load(checkpoint, weights_only=True)
    saved['retention']['episodes'][0]['success'] = not saved['retention']['episodes'][0]['success']
    changed = tmp_path / 'changed.pt'
    torch.save(saved, changed)
    assert not replay_checkpoint(changed)['valid']


def test_study_source_snapshot_reconstructs_its_recorded_fingerprint(tmp_path):
    from aether.evidence.receipts import source_fingerprint

    output = tmp_path / 'study'
    result = run_study(small_config(), output)
    restored = tmp_path / 'source'
    with zipfile.ZipFile(output / 'source-snapshot.zip') as archive:
        assert 'aether/live/web/app.js' in archive.namelist()
        assert 'aether/live/web/styles.css' in archive.namelist()
        assert 'aether/live/web/index.html' in archive.namelist()
        archive.extractall(restored)
    assert source_fingerprint(restored) == result['metadata']['source_fingerprint']


@pytest.mark.parametrize('task_family', ['straight', 'distraction', 'waypoint'])
def test_study_replays_selected_task_and_records_its_training_inputs(tmp_path, task_family):
    from aether.environments.stress import stress_expert_batch
    from aether.kernel.runtime import TissueModel
    from aether.config import ModelConfig
    from torch.nn import functional as F

    config = small_config()
    config.task_family = task_family
    output = tmp_path / task_family
    # A wrong support-task route changes the first loss on the actual expert
    # observations, even if a short-trained policy happens to choose the same actions.
    model = TissueModel(ModelConfig(**config.model), 'wave', seed=3)
    support = stress_expert_batch([300_000, 300_001], task_family=task_family,
                                 delay=3, horizon=9)
    logits = model(torch.from_numpy(support['observations']), disturbance_seed=3000)
    mask = torch.from_numpy(support['mask'])
    expected_loss = float(F.cross_entropy(logits[mask], torch.from_numpy(support['actions'])[mask]).detach())

    result = run_study(config, output)
    assert result['config']['task_family'] == task_family
    assert result['runs'][1]['training_curve'][0]['loss'] == expected_loss
    for run in result['runs']:
        path = output / run['checkpoint']
        saved = torch.load(path, weights_only=True)
        assert saved['task_family'] == task_family
        assert replay_checkpoint(path)['valid']


def test_replay_defaults_legacy_checkpoint_to_straight_task(tmp_path):
    result = run_study(small_config(), tmp_path / 'study')
    checkpoint = tmp_path / 'study' / result['runs'][1]['checkpoint']
    saved = torch.load(checkpoint, weights_only=True)
    saved.pop('task_family', None)
    legacy = tmp_path / 'legacy.pt'
    torch.save(saved, legacy)
    assert replay_checkpoint(legacy)['valid']


def test_evaluation_executes_distraction_and_waypoint_contracts():
    from aether.experiments.study import evaluate

    class CueFollower(torch.nn.Module):
        """Keep the initial direction, but visibly respond to later cue pulses."""
        def state_digest(self):
            return 'fixed-policy'

        def initial_state(self, batch_size):
            return torch.zeros(batch_size, 6)

        def step(self, observation, state, **kwargs):
            cue = observation[:, :6]
            state = torch.where(cue.sum(-1, keepdim=True) > 0, cue, state)
            return torch.cat((state, state.new_zeros(state.shape[0], 1)), -1), state

    model = CueFollower()
    straight = evaluate(model, [11], delay=3, horizon=9)
    distraction = evaluate(model, [11], delay=3, horizon=9, task_family='distraction')
    waypoint = evaluate(model, [11], delay=3, horizon=9, task_family='waypoint')
    assert straight['episodes'][0]['success'] is True
    assert distraction['episodes'][0]['actions'][1] != straight['episodes'][0]['actions'][1]
    assert waypoint['episodes'][0]['actions'] != straight['episodes'][0]['actions']


@pytest.mark.parametrize('kwargs', [
    {'updates': 0}, {'seeds': [1, 1]}, {'seeds': [-1]}, {'seeds': [10001]},
    {'protocols': ['fiction']}, {'batch_size': True}, {'learning_rate': float('nan')},
    {'protocols': ['wave']}, {'retention_delay': 2},
    {'task_family': 'unknown'}, {'task_family': []},
    {'task_family': 'waypoint', 'horizon': 5},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        StudyConfig(**kwargs)
