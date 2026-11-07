import dataclasses

import numpy as np
import pytest

from aether.neurowave import ChannelConfig, NeuralPacket, PacketBus


def packet(sender=0, receiver=1, tick=0, value=1, **kwargs):
    fields = dict(sender_id=sender, receiver_id=receiver, timestep=tick,
                  mode_count=4, payload=np.full((4, 8, 2), value, dtype=np.float16),
                  precision_bits=16, topology_version=0)
    fields.update(kwargs)
    return NeuralPacket(**fields)


def test_payload_is_immutable_owns_snapshot_and_counts_actual_wire_bytes():
    source = np.ones((4, 8, 2), dtype=np.float16)
    item = packet(payload=source)
    source.fill(7)
    assert np.all(item.payload == 1)
    assert item.payload_bytes == 128
    assert len(item.wire_bytes) == 128
    with pytest.raises(ValueError):
        item.payload[0, 0, 0] = 2
    with pytest.raises(ValueError):
        item.payload.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.sender_id = 7


def test_payload_metadata_changes_cannot_change_packet_shape_or_wire_format():
    item = packet()
    exposed = item.payload
    exposed.shape = (64,)
    exposed.dtype = np.uint8
    assert item.payload.shape == (4, 8, 2)
    assert item.payload.dtype == np.float16
    assert item.payload_bytes == 128


def test_bus_snapshots_links_and_rejects_cross_source_budget_overrun():
    links = [(0, 1), (1, 0)]
    bus = PacketBus(links, ChannelConfig(payload_budget_bytes=128))
    links.append((0, 2))
    bus.send([packet()], sender_id=0)
    with pytest.raises(ValueError):
        bus.send([packet(sender=1, receiver=0)], sender_id=1)
    assert bus.payload_bytes_by_tick == {0: 128}
    assert len(bus.deliver(0)) == 1
    with pytest.raises(ValueError):
        bus.send([packet(receiver=2, tick=1)], sender_id=0)


def test_invalid_recipient_filter_does_not_advance_delivery_clock():
    bus = PacketBus([(0, 1)])
    bus.send([packet()], sender_id=0)
    with pytest.raises(ValueError):
        bus.deliver(9, receiver_id=2)
    assert len(bus.deliver(0)) == 1


@pytest.mark.parametrize('fields', [
    {'sender_id': -1}, {'receiver_id': True}, {'timestep': 0.2},
    {'mode_count': 0}, {'topology_version': -1}, {'precision_bits': 8},
    {'payload': np.ones((4, 8), dtype=np.float16)},
    {'payload': np.ones((3, 8, 2), dtype=np.float16)},
    {'payload': np.ones((4, 8, 2), dtype=np.float32)},
    {'payload': np.full((4, 8, 2), np.nan, dtype=np.float16)},
])
def test_rejects_invalid_packet_fields(fields):
    with pytest.raises((TypeError, ValueError)):
        packet(**fields)


def test_clean_bus_direction_authority_order_and_payload_budget():
    bus = PacketBus([(0, 1), (1, 0)], ChannelConfig(payload_budget_bytes=256))
    bus.send([packet(sender=1, receiver=0)], sender_id=1)
    bus.send([packet()], sender_id=0)
    assert bus.payload_bytes_by_tick == {0: 256}
    assert [(p.receiver_id, p.sender_id) for p in bus.deliver(0)] == [(0, 1), (1, 0)]
    assert bus.deliver(0) == ()
    with pytest.raises(ValueError):
        bus.send([packet()], sender_id=0)
    with pytest.raises(ValueError):
        bus.send([packet(sender=1, receiver=0, tick=1)], sender_id=0)
    with pytest.raises(ValueError):
        bus.send([packet(sender=1, receiver=1, tick=1)], sender_id=1)


@pytest.mark.parametrize('bad', [
    lambda: packet(sender=1), lambda: packet(receiver=2),
    lambda: packet(topology_version=1),
    lambda: packet(mode_count=3, payload=np.ones((3, 8, 2), dtype=np.float16)),
    lambda: packet(payload=np.ones((4, 7, 2), dtype=np.float16)),
    lambda: packet(precision_bits=32, payload=np.ones((4, 8, 2), dtype=np.float32)),
    lambda: packet(tick=1),
])
def test_invalid_batch_preserves_queue_budget_clock_and_rng(bad):
    config = ChannelConfig(seed=81, noise_std=.2, phase_std=.1, latency_jitter=2)
    actual, control = [PacketBus([(0, 1)], config) for _ in range(2)]
    with pytest.raises(ValueError):
        actual.send([packet(), bad()], sender_id=0)
    assert actual.payload_bytes_by_tick == {}
    assert actual.pending_count == 0
    for bus in (actual, control):
        bus.send([packet()], sender_id=0)
    a, b = actual.deliver(3), control.deliver(3)
    assert len(a) == len(b) == 1
    np.testing.assert_array_equal(a[0].payload, b[0].payload)


def test_budget_rejection_is_atomic_and_counts_lost_packets():
    bus = PacketBus([(0, 1), (0, 2)], ChannelConfig(payload_budget_bytes=128, loss_probability=1))
    with pytest.raises(ValueError):
        bus.send([packet(), packet(receiver=2)], sender_id=0)
    bus.send([packet()], sender_id=0)
    assert bus.payload_bytes_by_tick == {0: 128}
    assert bus.deliver(0) == ()


def test_latency_filtering_and_tick_order():
    bus = PacketBus([(0, 1), (0, 2)], ChannelConfig(latency_ticks=2))
    bus.send([packet(), packet(receiver=2)], sender_id=0)
    assert bus.deliver(1) == ()
    assert len(bus.deliver(2, receiver_id=1)) == 1
    assert bus.pending_count == 1
    assert bus.deliver(2)[0].receiver_id == 2
    with pytest.raises(ValueError):
        bus.deliver(1)
    with pytest.raises(ValueError):
        bus.send([packet(tick=1)], sender_id=0)


def test_impairments_are_seeded_independently_and_finite():
    config = ChannelConfig(seed=4, noise_std=.2, phase_std=.3, crosstalk=.2,
                           latency_ticks=1, latency_jitter=2, loss_probability=.1,
                           congestion_probability=.1)
    outputs = []
    for _ in range(2):
        bus = PacketBus([(0, 1), (0, 2)], config)
        for tick in range(10):
            bus.send([packet(tick=tick), packet(receiver=2, tick=tick, value=2)], sender_id=0)
        outputs.append([(p.sender_id, p.receiver_id, p.timestep, p.wire_bytes) for p in bus.deliver(20)])
    assert outputs[0] == outputs[1]
    assert outputs[0]
    assert any(raw != packet().wire_bytes for _, _, _, raw in outputs[0])
    # Enabling phase draws must not change independent latency/loss decisions.
    a = PacketBus([(0, 1)], ChannelConfig(seed=33, loss_probability=.3, latency_jitter=5))
    b = PacketBus([(0, 1)], ChannelConfig(seed=33, loss_probability=.3, latency_jitter=5, phase_std=.4))
    for tick in range(10):
        for bus in (a, b):
            bus.send([packet(tick=tick)], sender_id=0)
    for tick in range(16):
        assert [p.timestep for p in a.deliver(tick)] == [p.timestep for p in b.deliver(tick)]


def test_crosstalk_mixes_other_packets_and_phase_preserves_amplitude():
    bus = PacketBus([(0, 1), (0, 2)], ChannelConfig(crosstalk=.25))
    bus.send([packet(value=0), packet(receiver=2, value=4)], sender_id=0)
    out = bus.deliver(0)
    np.testing.assert_array_equal(out[0].payload, np.ones((4, 8, 2)))
    np.testing.assert_array_equal(out[1].payload, np.full((4, 8, 2), 3))
    bus = PacketBus([(0, 1)], ChannelConfig(seed=8, phase_std=1))
    bus.send([packet()], sender_id=0)
    np.testing.assert_allclose(np.linalg.norm(bus.deliver(0)[0].payload.astype(float), axis=-1), np.sqrt(2), atol=.001)


@pytest.mark.parametrize('fields', [
    {'loss_probability': -1}, {'congestion_probability': 2}, {'noise_std': np.nan},
    {'phase_std': -1}, {'crosstalk': 2}, {'latency_ticks': 1.1},
    {'latency_jitter': -1}, {'payload_budget_bytes': 0}, {'seed': True},
    {'latency_jitter': 2**64},
])
def test_channel_configuration_is_validated(fields):
    with pytest.raises((TypeError, ValueError)):
        ChannelConfig(**fields)
