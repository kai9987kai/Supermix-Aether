"""Executed paired worlds and immutable, measured live snapshots."""
import importlib
import json

import pytest
import torch

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel


def model(protocol='wave', **changes):
    values = dict(neuron_count=16, scalar_edges=32, region_count=4,
        region_message_edges=8, state_dimension=4, mode_count=3, complex_dimensions=2,
        initial_residual_gain=0.7)
    values.update(changes)
    return TissueModel(ModelConfig(**values), protocol, 11)


def session(**settings):
    LiveSession = importlib.import_module('aether.live.session').LiveSession
    return LiveSession(model(), **settings)


def test_clean_pair_is_identical_and_query_never_updates_weights():
    live = session()
    digest = live.model.state_digest()
    for _ in range(9):
        frame = live.step()
        assert frame['candidate'] == frame['control']
        assert frame['comparison']['effective_action_differences'] == 0
    assert live.model.state_digest() == digest
    assert frame['done']
    next_frame = live.step()
    assert next_frame['seed'] == frame['seed'] + 1
    assert next_frame['episode'] == frame['episode'] + 1
    assert next_frame['tick'] == 1


def test_snapshots_are_copies_and_do_not_run_inference():
    provenance = {'kind': 'fresh', 'nested': {'name': 'test'}}
    live = session(provenance=provenance)
    provenance['nested']['name'] = 'changed'
    before = live.snapshot()
    assert before['tick'] == 0 and before['candidate']['action'] is None
    live.model.step = lambda *a, **k: (_ for _ in ()).throw(AssertionError('inference'))
    returned = live.snapshot()
    returned['settings']['delay'] = 99
    returned['candidate']['position'][0] = 99
    assert live.snapshot() == before
    assert before['provenance']['nested']['name'] == 'test'


@pytest.mark.parametrize('settings', [dict(loss_probability=float('nan')), dict(delay=True),
    dict(latency_ticks=1.5), dict(noise_std=-1), dict(unknown=1), dict(intervention='wrong')])
def test_invalid_configuration_is_atomic_including_channel_rng(settings):
    a, b = session(), session()
    a.configure(noise_std=0.3, loss_probability=0.4)
    b.configure(noise_std=0.3, loss_probability=0.4)
    assert a.step() == b.step()
    before = a.snapshot()
    with pytest.raises(ValueError):
        a.configure(**settings)
    assert a.snapshot() == before
    assert a.step() == b.step()


def test_settings_restart_same_seed_and_replacement_starts_new_provenance_epoch():
    live = session(seed=90)
    live.step()
    changed = live.configure(delay=5, bandwidth_fraction=0.5)
    assert changed['tick'] == 0 and changed['seed'] == 90
    assert changed['history'] == []
    assert live.horizon == 11
    assert changed['candidate'] == changed['control']
    replaced = live.replace_model(model('vector'), {'kind': 'session-adapted'})
    assert replaced['tick'] == 0 and replaced['model']['protocol'] == 'vector'
    assert replaced['provenance']['kind'] == 'session-adapted'
    assert replaced['history'] == []
    with pytest.raises(ValueError):
        live.reset(seed=-1)
    assert live.snapshot() == replaced


def test_telemetry_is_from_executed_state_and_delivered_packets():
    live = session()
    frame = live.step()
    branch = frame['candidate']
    assert len(branch['probabilities']) == 7
    assert sum(branch['probabilities']) == pytest.approx(1)
    assert len(branch['links']) == 8 and len(branch['mode_energy']) == 3
    assert branch['payload_bytes'] == 192
    for row in branch['regions']:
        state = live.candidate_state[:, live.model.neuron_regions == row['id']]
        assert row['activity'] == pytest.approx(float(state.square().mean().sqrt()))
    live.configure(loss_probability=1)
    frame = live.step()
    assert frame['candidate']['attempted_payload_bytes'] == 192
    assert frame['candidate']['admitted_payload_bytes'] == 192
    assert frame['candidate']['surviving_payload_bytes'] == 0
    assert frame['candidate']['delivered_payload_bytes'] == 0
    assert all(link['energy'] == 0 for link in frame['candidate']['links'])
    assert any(link['energy'] > 0 for link in frame['control']['links'])


@pytest.mark.parametrize('protocol,zero_links', [('scalar', False), ('wave', True)])
def test_scalar_and_empty_wire_telemetry_stays_finite(protocol, zero_links):
    LiveSession = importlib.import_module('aether.live.session').LiveSession
    live = LiveSession(model(protocol, **({'region_message_edges': 0} if zero_links else {})))
    frame = live.step()
    assert frame['candidate']['payload_bytes'] == 0
    assert all(value == 0 for value in frame['candidate']['mode_energy'])
    json.dumps(frame, allow_nan=False)


def test_trace_is_bounded_and_export_labels_exploratory_truncation():
    live = session()
    for _ in range(520):
        live.step()
    trace = live.export()
    assert trace['exploratory'] is True
    assert trace['full_neural_state_replay'] is False
    assert trace['truncated'] is True
    assert len(trace['frames']) <= 512
    assert trace['frame']['model']['digest'] == live.model.state_digest()
    assert len(live.snapshot()['history']) <= 512
    json.dumps(trace, allow_nan=False)


def test_disturbances_do_not_consume_global_rng_and_only_one_inference_per_branch_tick():
    live = session()
    live.configure(noise_std=0.5, intervention='random_phase')
    original_step = live.model.step
    calls = []
    def counting_step(*args, **kwargs):
        calls.append(1)
        return original_step(*args, **kwargs)
    live.model.step = counting_step
    before = torch.random.get_rng_state().clone()
    live.step()
    live.snapshot()
    live.snapshot()
    assert len(calls) == 2
    assert torch.equal(torch.random.get_rng_state(), before)


def test_waiting_disagreements_are_not_counted_as_effective_actions():
    live = session(delay=3)
    live.configure(noise_std=1, intervention='random_phase')
    # Amplify the actual auxiliary route to expose intervention sensitivity.
    with torch.no_grad():
        live.model.residual_gain.fill_(100)
    frame = live.step()
    assert frame['candidate']['position'] == frame['control']['position'] == [0, 0, 0]
    assert frame['comparison']['effective_action_differences'] == 0
    assert frame['history'][-1]['movement_enabled'] is False
    assert frame['comparison']['waiting_action_differences'] == int(
        frame['candidate']['action'] != frame['control']['action'])
