"""Behavioral contracts for the trainable CPU tissue runtime."""
import dataclasses
import importlib

import numpy as np
import pytest
import torch


def api():
    config = importlib.import_module('aether.config')
    runtime = importlib.import_module('aether.kernel.runtime')
    return config.ModelConfig, runtime.TissueModel


def small(**changes):
    Config, _ = api()
    values = dict(neuron_count=16, scalar_edges=48, region_count=4,
                  region_message_edges=8, state_dimension=6, mode_count=3,
                  complex_dimensions=2)
    values.update(changes)
    return Config(**values)


def test_default_topology_has_authoritative_directed_edge_counts():
    Config, Model = api()
    model = Model(Config(), 'wave', 2)
    assert model.scalar_sources.shape == (4096,)
    assert model.region_sources.shape == (96,)
    assert len(set(zip(model.scalar_sources.tolist(), model.scalar_targets.tolist()))) == 4096
    assert len(set(zip(model.region_sources.tolist(), model.region_targets.tolist()))) == 96
    assert model.resource_metrics()['payload_bytes_per_tick'] == 12288
    topology = model.topology
    with pytest.raises(ValueError):
        topology.scalar_sources[0] = 99


@pytest.mark.parametrize('protocol', ['scalar_matched', 'vector', 'attention', 'phase', 'modes', 'wave'])
def test_zero_gain_is_bit_identical_to_scalar_with_shared_initialization(protocol):
    _, Model = api()
    base = Model(small(), 'scalar', 31)
    auxiliary = Model(small(), protocol, 31)
    observations = torch.randn(2, 5, 12, generator=torch.Generator().manual_seed(9))
    assert torch.equal(base(observations), auxiliary(observations))
    with torch.no_grad():
        auxiliary.residual_gain.fill_(0.7)
    assert torch.equal(base(observations), auxiliary(observations, intervention='zero_gain'))
    assert torch.equal(base(observations), auxiliary(observations, intervention='erase_packets'))


def test_all_auxiliary_protocols_have_equal_parameter_shapes_and_payloads():
    _, Model = api()
    models = [Model(small(), arm, 1) for arm in ['scalar_matched','vector','attention','phase','modes','wave']]
    signatures = [{name: tuple(p.shape) for name, p in m.named_parameters()} for m in models]
    assert all(s == signatures[0] for s in signatures)
    assert len({m.resource_metrics()['payload_bytes_per_tick'] for m in models}) == 1


@pytest.mark.parametrize('protocol', ['scalar','scalar_matched','vector','attention','phase','modes','wave'])
def test_executed_sequence_has_finite_gradients_and_trains(protocol):
    _, Model = api()
    model = Model(small(initial_residual_gain=0.2), protocol, 4)
    observations = torch.randn(3, 4, 12, generator=torch.Generator().manual_seed(6))
    before = model.state_digest()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    logits = model(observations)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), torch.arange(12) % 7)
    loss.backward()
    for name in ['edge_weights', 'input_encoder.weight', 'output_decoder.weight']:
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    if protocol != 'scalar':
        assert model.message_encoder.weight.grad.abs().sum() > 0
    optimizer.step()
    assert model.state_digest() != before


def test_replay_disturbance_and_query_state_immutability():
    _, Model = api()
    model = Model(small(initial_residual_gain=0.6), 'wave', 11)
    observations = torch.randn(2, 6, 12, generator=torch.Generator().manual_seed(3))
    before = model.state_digest()
    a = model(observations, intervention='random_phase', disturbance_seed=92)
    b = model(observations, intervention='random_phase', disturbance_seed=92)
    c = model(observations, intervention='random_phase', disturbance_seed=93)
    assert torch.equal(a,b)
    assert not torch.equal(a,c)
    assert model.state_digest() == before
    copy = Model(small(initial_residual_gain=0.6), 'wave', 21)
    copy.load_state_dict(model.state_dict())
    assert torch.equal(a, copy(observations, intervention='random_phase', disturbance_seed=92))


def test_modes_ignore_phase_and_phase_ignores_carrier_modes():
    _, Model = api()
    observations = torch.randn(1, 4, 12, generator=torch.Generator().manual_seed(6))
    modes = Model(small(initial_residual_gain=0.7), 'modes', 3)
    assert torch.equal(modes(observations), modes(observations, intervention='random_phase', disturbance_seed=9))
    phase = Model(small(initial_residual_gain=0.7), 'phase', 3)
    a = phase(observations)
    with torch.no_grad():
        phase.carriers.mul_(-1)
    assert torch.equal(a, phase(observations))


def test_numpy_reference_matches_independent_scalar_step():
    _, Model = api()
    ref = importlib.import_module('aether.kernel.reference').scalar_step
    model = Model(small(), 'scalar', 7)
    observation = np.random.default_rng(14).normal(size=(2,12)).astype(np.float32)
    state = np.random.default_rng(15).normal(size=(2,16,6)).astype(np.float32)
    logits, next_state = model.step(torch.from_numpy(observation), torch.from_numpy(state))
    expected_logits, expected_state = ref(model, observation, state)
    np.testing.assert_allclose(next_state.detach().numpy(), expected_state, atol=3e-7, rtol=3e-6)
    np.testing.assert_allclose(logits.detach().numpy(), expected_logits, atol=3e-7, rtol=3e-6)


def test_invalid_configuration_and_interventions_fail_closed():
    Config, Model = api()
    for changes in [dict(neuron_count=0), dict(region_count=17), dict(scalar_edges=1000), dict(mode_count=5), dict(initial_residual_gain=float('nan'))]:
        with pytest.raises(ValueError):
            small(**changes)
    model = Model(small(), 'wave', 1)
    with pytest.raises(ValueError):
        model(torch.zeros(1,2,12), intervention='typo')
    with pytest.raises(ValueError):
        model.step(torch.zeros(1,11), model.initial_state(1))
    assert dataclasses.asdict(Config())['initial_residual_gain'] == 0.0


def test_encoded_payload_is_fixed_fp16_wire_shape_and_uses_straight_through_gradient():
    _, Model = api()
    model = Model(small(initial_residual_gain=0.3), 'wave', 13)
    state = torch.randn(2,16,6, generator=torch.Generator().manual_seed(8), requires_grad=True)
    payload = model.encoded_payload(state)
    assert payload.shape == (2,8,3,2,2)
    assert torch.equal(payload, payload.half().float())
    payload.square().sum().backward()
    assert state.grad.abs().sum() > 0
    assert model.message_encoder.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(model.encoded_payload(state, intervention='erase_packets')) == 0


def test_attention_routing_is_learned_and_changes_received_dynamics():
    _, Model = api()
    model = Model(small(initial_residual_gain=0.9), 'attention', 9)
    observation = torch.randn(1,12, generator=torch.Generator().manual_seed(3))
    state = torch.randn(1,16,6, generator=torch.Generator().manual_seed(5))
    a, _ = model.step(observation, state)
    a.square().sum().backward()
    assert model.routing_query.weight.grad.abs().sum() > 0
    assert model.routing_key.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.routing_query.weight.mul_(20)
    b, _ = model.step(observation, state)
    assert not torch.equal(a,b)


def test_auxiliary_residual_is_clipped_and_added_after_base_activation():
    _, Model = api()
    model = Model(small(initial_residual_gain=0.5), 'vector', 2)
    with torch.no_grad():
        model.input_encoder.weight.zero_()
        model.input_encoder.bias.zero_()
        model.neuron_bias.zero_()
        model.message_encoder.weight.zero_()
        model.message_encoder.bias.fill_(10)
        model.message_decoder.weight.fill_(10)
    _, state = model.step(torch.zeros(1,12), model.initial_state(1))
    # F_base is exactly zero; received values exceed the fixed +/-1 bound.
    assert torch.equal(state, torch.full_like(state, 0.5))


def test_observation_enters_only_sensory_region_then_propagates_to_readout():
    Config, Model = api()
    config = Config(neuron_count=4, scalar_edges=4, region_count=2,
                   region_message_edges=2, state_dimension=3, mode_count=1,
                   complex_dimensions=1)
    model = Model(config, 'scalar', 3)
    state = model.initial_state(1)
    a, cue_state = model.step(torch.ones(1,12), state)
    b, blank_state = model.step(torch.zeros(1,12), state)
    assert not torch.equal(cue_state[:, :2], blank_state[:, :2])
    assert torch.equal(cue_state[:, 2:], blank_state[:, 2:])
    assert torch.equal(a,b)
    # Ring edge 1->2 carries sensory state to the first readout neuron.
    after_cue, propagated = model.step(torch.zeros(1,12), cue_state)
    after_blank, _ = model.step(torch.zeros(1,12), blank_state)
    assert not torch.equal(after_cue, after_blank)
    expected = model.output_decoder(propagated[:, 2:].mean(1))
    assert torch.equal(after_cue, expected)


def test_single_region_can_both_receive_observation_and_read_out():
    Config, Model = api()
    config = Config(neuron_count=3, scalar_edges=3, region_count=1,
                   region_message_edges=0, state_dimension=2, mode_count=1,
                   complex_dimensions=1)
    model = Model(config, 'scalar', 4)
    blank = model(torch.zeros(1,2,12))
    observed = model(torch.ones(1,2,12))
    assert not torch.equal(blank, observed)


@pytest.mark.parametrize('protocol', ['attention', 'phase', 'wave'])
def test_fixed_wire_decode_uses_only_receiver_local_state(protocol):
    Config, Model = api()
    config = Config(neuron_count=8, scalar_edges=8, region_count=4,
                   region_message_edges=4, state_dimension=3, mode_count=2,
                   complex_dimensions=2)
    model = Model(config, protocol, 9)
    state = torch.randn(1,8,3, generator=torch.Generator().manual_seed(1))
    payload = model.encoded_payload(state).detach()
    before = model.receiver_decode(payload, state)
    changed = state.clone()
    changed[:, 4:6] += 3  # Region2 sends to region3, but receiver3 is unchanged.
    after = model.receiver_decode(payload, changed)
    assert torch.equal(before[:, 6:8], after[:, 6:8])


@pytest.mark.parametrize('protocol', ['phase', 'wave'])
def test_phase_wire_reserves_first_mode_for_quantized_unit_pilots(protocol):
    _, Model = api()
    model = Model(small(), protocol, 8)
    state = torch.randn(1,16,6, generator=torch.Generator().manual_seed(3))
    payload = model.encoded_payload(state)
    norm = payload[:, :, 0].square().sum(-1)
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=0.001, rtol=0)


def test_loading_checkpoint_restores_authoritative_topology_and_rejects_stale_derivations():
    _, Model = api()
    source = Model(small(), 'wave', 8)
    restored = Model(small(), 'wave', 9)
    restored.load_state_dict(source.state_dict())
    assert np.array_equal(restored.topology.scalar_sources, source.topology.scalar_sources)
    assert np.array_equal(restored.topology.region_sources, source.topology.region_sources)
    corrupted = {name: value.clone() for name, value in source.state_dict().items()}
    corrupted['pooling'][0, 0] += 1
    with pytest.raises(ValueError):
        restored.load_state_dict(corrupted)


@pytest.mark.parametrize('assign', [False, True])
def test_failed_checkpoint_load_preserves_tensors_and_authoritative_topology(assign):
    _, Model = api()
    source = Model(small(), 'wave', 11)
    restored = Model(small(), 'wave', 21)
    before = restored.state_digest()
    topology = restored.topology
    malformed = {name: value.clone() for name, value in source.state_dict().items()}
    malformed['output_decoder.weight'] = malformed['output_decoder.weight'][:1]
    with pytest.raises(RuntimeError):
        restored.load_state_dict(malformed, assign=assign)
    assert restored.state_digest() == before
    assert restored.topology is topology


@pytest.mark.parametrize('corruption', ['missing', 'unexpected', 'non_tensor', 'dtype'])
def test_checkpoint_preflight_rejects_malformed_mapping_without_mutation(corruption):
    _, Model = api()
    source = Model(small(), 'wave', 11)
    restored = Model(small(), 'wave', 21)
    before = restored.state_digest()
    malformed = {name: value.clone() for name, value in source.state_dict().items()}
    if corruption == 'missing':
        del malformed['output_decoder.weight']
    elif corruption == 'unexpected':
        malformed['unknown_tensor'] = torch.ones(1)
    elif corruption == 'non_tensor':
        malformed['output_decoder.weight'] = 'invalid'
    else:
        malformed['output_decoder.weight'] = malformed['output_decoder.weight'].double()
    with pytest.raises((RuntimeError, TypeError)):
        restored.load_state_dict(malformed)
    assert restored.state_digest() == before
