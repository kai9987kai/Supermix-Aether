import json

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


@pytest.mark.parametrize('kwargs', [
    {'updates': 0}, {'seeds': [1, 1]}, {'seeds': [-1]}, {'seeds': [10001]},
    {'protocols': ['fiction']}, {'batch_size': True}, {'learning_rate': float('nan')},
    {'protocols': ['wave']}, {'retention_delay': 2},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        StudyConfig(**kwargs)
