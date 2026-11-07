"""Independent packet delivery and NumPy decoding of clean tensor transport.

This checks the numerical fast path against the validated reference bus. It
does not claim differentiable equivalence for reference-only impairments.
"""

import math

import numpy as np
import torch

from aether.neurowave import ChannelConfig, NeuralPacket, PacketBus


def _array(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


def check_transport_conformance(model) -> dict:
    if model.protocol == 'scalar':
        return {'valid': True, 'payload_bytes_per_tick': 0, 'max_absolute_error': 0.0,
                'scope': 'scalar-only; no packet route'}
    c = model.config
    if c.region_message_edges == 0:
        return {'valid': True, 'payload_bytes_per_tick': 0, 'max_absolute_error': 0.0,
                'scope': 'no directed message links'}
    state = torch.from_numpy(np.random.default_rng(91).normal(
        0, .1, (1, c.neuron_count, c.state_dimension)).astype(np.float32))
    digest = model.state_digest()
    with torch.no_grad():
        payloads = model.encoded_payload(state)[0].numpy().astype(np.float16)
        actual = model._messages(state, 'none', 0).numpy()[0]
    sources, targets = model.topology.region_sources, model.topology.region_targets
    budget = c.region_message_edges * c.mode_count * c.complex_dimensions * 2 * 2
    links = list(zip(sources.tolist(), targets.tolist()))
    bus = PacketBus(links, ChannelConfig(payload_budget_bytes=budget),
                    mode_count=c.mode_count, complex_dim=c.complex_dimensions)
    for sender in sorted(set(sources.tolist())):
        packets = [NeuralPacket(sender, int(targets[i]), 0, c.mode_count, payloads[i], 16, 0)
                   for i in range(len(links)) if sources[i] == sender]
        bus.send(packets, sender_id=sender)
    decoded_packets = bus.deliver(0)
    if len(decoded_packets) != len(links):
        raise RuntimeError('clean reference bus lost a packet')
    pooled = _array(model.pooling) @ _array(state)[0]
    weights = 1 / _array(model.region_degree)[targets]
    if model.protocol == 'attention':
        query = pooled @ _array(model.routing_query.weight).T
        packet_features = payloads.astype(np.float64).reshape(len(links), -1) @ _array(model.message_decoder.weight).T
        key = packet_features @ _array(model.routing_key.weight).T
        scores = (query[targets] * key).sum(-1) / math.sqrt(c.state_dimension)
        weights = np.zeros_like(scores)
        for receiver in range(c.region_count):
            mask = targets == receiver
            if mask.any():
                exponential = np.exp(scores[mask] - scores[mask].max())
                weights[mask] = exponential / exponential.sum()
    if model.protocol in ('phase', 'wave'):
        phases = math.pi * np.tanh(pooled @ _array(model.phase_encoder.weight).T + _array(model.phase_encoder.bias))
        pilot = payloads.astype(np.float64)[:, 0]
        sender_phases = np.arctan2(pilot[..., 1], pilot[..., 0])
        coherence = (1 + np.cos(sender_phases - phases[targets]).mean(-1)) * .5
        weights = weights * coherence
    lookup = {pair: index for index, pair in enumerate(links)}
    delivered = np.zeros((c.region_count, c.mode_count, c.complex_dimensions, 2))
    payload_identity = True
    for packet in decoded_packets:
        index = lookup[(packet.sender_id, packet.receiver_id)]
        payload_identity &= np.array_equal(packet.payload, payloads[index])
        delivered[packet.receiver_id] += packet.payload.astype(np.float64) * weights[index]
    if model.protocol in ('modes', 'wave'):
        delivered *= _array(model.carriers)[:, :, None, None]
    decoded = delivered.reshape(c.region_count, -1) @ _array(model.message_decoder.weight).T
    expected = decoded[model.topology.neuron_regions]
    error = float(np.max(np.abs(actual - expected)))
    valid = bool(payload_identity and error < 1e-6 and model.state_digest() == digest)
    return {'valid': valid, 'payload_bytes_per_tick': budget, 'max_absolute_error': error,
            'scope': 'clean FP16 wire payload, directed delivery and NumPy receiver decoding'}
