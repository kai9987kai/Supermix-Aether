"""Actual packet delivery, seeded disturbances, and bounded delay."""
import importlib

import pytest
import torch


def channel(**settings):
    Channel = importlib.import_module('aether.live.channel').PacketChannel
    return Channel(seed=31, **settings)


def packets():
    return torch.arange(32, dtype=torch.float32).reshape(1, 4, 2, 2, 2).half().float()


def test_clean_wire_is_exact_identity_and_accounts_fp16_bytes():
    wire = channel()
    value = packets()
    assert wire(value) is value
    assert wire.attempted_payload_bytes == wire.admitted_payload_bytes == 64
    assert torch.equal(wire.last_transmitted, value)
    assert torch.equal(wire.last_delivered, value)


def test_seeded_loss_noise_bandwidth_are_real_deterministic_wire_operations():
    a = channel(loss_probability=0.2, noise_std=0.4, bandwidth_fraction=0.75)
    b = channel(loss_probability=0.2, noise_std=0.4, bandwidth_fraction=0.75)
    original = packets()
    for _ in range(4):
        output = a(original)
        assert torch.equal(output, b(original))
        assert torch.equal(output, output.half().float())
        assert a.admitted_payload_bytes <= 48
        assert torch.count_nonzero(output) < 32
    assert torch.equal(original, packets())


def test_latency_delivers_previous_wire_and_queue_stays_bounded():
    wire = channel(latency_ticks=4)
    value = packets()
    for tick in range(4):
        assert torch.count_nonzero(wire(value + tick)) == 0
        assert wire.delivered_payload_bytes == 0
    assert torch.equal(wire(value + 4), value)
    assert wire.delivered_payload_bytes == 64
    for _ in range(100):
        wire(value)
    assert len(wire.queue) <= 4


@pytest.mark.parametrize('changes', [dict(latency_ticks=5), dict(loss_probability=float('nan')),
    dict(bandwidth_fraction=-1), dict(noise_std=True)])
def test_invalid_channel_settings_rejected(changes):
    with pytest.raises(ValueError):
        channel(**changes)


def test_packet_transform_is_used_by_runtime_and_default_is_identical():
    from aether.config import ModelConfig
    from aether.kernel.runtime import TissueModel
    model = TissueModel(ModelConfig(neuron_count=8, scalar_edges=8, region_count=2,
        region_message_edges=2, state_dimension=3, mode_count=2, complex_dimensions=1,
        initial_residual_gain=0.8), 'wave', 1)
    observation, state = torch.ones(1, 12), model.initial_state(1)
    clean = model.step(observation, state)
    identity = model.step(observation, state, packet_transform=channel())
    assert all(torch.equal(a, b) for a, b in zip(clean, identity))
    erased = model.step(observation, state, packet_transform=channel(loss_probability=1))
    zero = model.step(observation, state, intervention='erase_packets')
    assert all(torch.equal(a, b) for a, b in zip(erased, zero))
    assert not torch.equal(clean[1], erased[1])
