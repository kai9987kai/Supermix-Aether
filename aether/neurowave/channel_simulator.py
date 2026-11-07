"""Seeded CPU packet reference transport with atomic admission."""

from dataclasses import dataclass, replace
import numbers

import numpy as np

from .protocol import NeuralPacket, _integer


@dataclass(frozen=True)
class ChannelConfig:
    """Reference-channel settings.

    Loss and congestion independently erase packets. Gaussian noise is applied
    to real pairs after crosstalk (convex mixing with other batch packets) and
    per-complex-component Gaussian phase rotation. Latency is fixed ticks plus
    uniform integer jitter in [0, latency_jitter]. Each disturbance owns an
    independent SeedSequence-derived RNG. Budget counts admitted attempted
    payload bytes, including erased packets; headers/queues are excluded.
    """

    seed: int = 0
    loss_probability: float = 0.0
    noise_std: float = 0.0
    crosstalk: float = 0.0
    latency_ticks: int = 0
    latency_jitter: int = 0
    phase_std: float = 0.0
    congestion_probability: float = 0.0
    payload_budget_bytes: int = 12288

    def __post_init__(self):
        for name in ('seed', 'latency_ticks', 'latency_jitter'):
            _integer(getattr(self, name), name)
        _integer(self.payload_budget_bytes, 'payload_budget_bytes', 1)
        if self.latency_jitter >= np.iinfo(np.int64).max:
            raise ValueError('latency_jitter exceeds the integer sampler range')
        for name in ('loss_probability', 'congestion_probability', 'crosstalk', 'noise_std', 'phase_std'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            if name in ('loss_probability', 'congestion_probability', 'crosstalk') and value > 1:
                raise ValueError(f'{name} must be <= 1')


class PacketBus:
    """PacketBus(links, config=None, *, topology_version=0, mode_count=4,
    complex_dim=8, precision_bits=16).

    ``links`` is an immutable snapshot of allowed directed (sender, receiver)
    pairs. ``send(packets, sender_id=...)`` authenticates a nonempty same-source,
    same-tick batch using the externally trusted sender_id. Call it once per
    sender per tick, in nondecreasing tick order. Budgets aggregate all senders.
    Duplicate directed messages in one tick are forbidden. Invalid batches
    change no clocks, budget, queue or random stream. ``deliver(timestep,
    receiver_id=None)`` drains due packets (optionally one recipient), ordered
    by receiver, sender and original tick. Receive ticks cannot move backwards.

    Config/topology are fixed for this bus's lifetime; create another bus after
    a topology change. This is a single-threaded reference implementation.
    """

    def __init__(self, links, config=None, *, topology_version=0, mode_count=4,
                 complex_dim=8, precision_bits=16):
        self._config = config if config is not None else ChannelConfig()
        if not isinstance(self._config, ChannelConfig):
            raise TypeError('config must be ChannelConfig')
        self._topology_version = _integer(topology_version, 'topology_version')
        self._mode_count = _integer(mode_count, 'mode_count', 1)
        self._complex_dim = _integer(complex_dim, 'complex_dim', 1)
        self._precision_bits = _integer(precision_bits, 'precision_bits', 1)
        if precision_bits not in (16, 32, 64):
            raise ValueError('precision_bits must be 16, 32 or 64')
        normalized = []
        for edge in links:
            if len(edge) != 2:
                raise ValueError('each link must contain sender and receiver')
            normalized.append(tuple(_integer(x, 'link endpoint') for x in edge))
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError('links must be nonempty and unique')
        self._links = frozenset(normalized)
        self._regions = frozenset(x for pair in self._links for x in pair)
        streams = np.random.SeedSequence(self._config.seed).spawn(5)
        self._loss_rng, self._noise_rng, self._latency_rng, self._phase_rng, self._congestion_rng = [np.random.default_rng(s) for s in streams]
        self._queue = []
        self._bytes_by_tick = {}
        self._sender_ticks = {}
        self._last_send_tick = -1
        self._last_receive_tick = -1

    @property
    def links(self):
        return self._links

    @property
    def config(self):
        return self._config

    @property
    def payload_bytes_by_tick(self):
        return dict(self._bytes_by_tick)

    @property
    def pending_count(self):
        return len(self._queue)

    def send(self, packets, *, sender_id):
        sender_id = _integer(sender_id, 'sender_id')
        packets = tuple(packets)
        if not packets:
            raise ValueError('send requires a nonempty packet batch')
        if not all(isinstance(p, NeuralPacket) for p in packets):
            raise TypeError('all packets must be NeuralPacket')
        # Snapshot and revalidate fields before any mutation, including forged
        # dataclass instances or an original ndarray modified by its owner.
        packets = tuple(replace(p) for p in packets)
        tick = packets[0].timestep
        if tick < max(self._last_send_tick, self._last_receive_tick):
            raise ValueError('send timestep moved backwards')
        if tick <= self._sender_ticks.get(sender_id, -1):
            raise ValueError('sender already submitted this tick')
        recipients = set()
        for packet in packets:
            if packet.sender_id != sender_id:
                raise ValueError('sender authority mismatch')
            if packet.timestep != tick:
                raise ValueError('all batch packets must have the same timestep')
            if (packet.sender_id, packet.receiver_id) not in self._links:
                raise ValueError('packet uses a forbidden directed link')
            if packet.topology_version != self._topology_version:
                raise ValueError('topology version mismatch')
            if packet.mode_count != self._mode_count or packet.payload.shape != (self._mode_count, self._complex_dim, 2):
                raise ValueError('packet shape mismatch')
            if packet.precision_bits != self._precision_bits:
                raise ValueError('packet precision mismatch')
            if packet.receiver_id in recipients:
                raise ValueError('duplicate directed message in a tick')
            recipients.add(packet.receiver_id)
        byte_count = sum(p.payload_bytes for p in packets)
        next_budget = self._bytes_by_tick.get(tick, 0) + byte_count
        if next_budget > self._config.payload_budget_bytes:
            raise ValueError('per-tick payload budget exceeded')
        # Canonical order makes a batch's stochastic outcome independent of
        # caller enumeration. Crosstalk uses original admitted batch payloads.
        packets = tuple(sorted(packets, key=lambda p: p.receiver_id))
        payloads = np.stack([p.payload.astype(np.float64) for p in packets])
        config = self._config
        if config.crosstalk and len(packets) > 1:
            payloads = (1 - config.crosstalk) * payloads + config.crosstalk * (payloads.sum(axis=0) - payloads) / (len(packets) - 1)
        for i, packet in enumerate(packets):
            lost = self._loss_rng.random() < config.loss_probability
            congested = self._congestion_rng.random() < config.congestion_probability
            jitter = int(self._latency_rng.integers(config.latency_jitter + 1)) if config.latency_jitter else 0
            payload = payloads[i]
            with np.errstate(over='ignore', invalid='ignore'):
                if config.phase_std:
                    phase = self._phase_rng.normal(0, config.phase_std, payload.shape[:-1])
                    phase = np.nan_to_num(phase)
                    real, imag = payload[..., 0], payload[..., 1]
                    payload = np.stack((real * np.cos(phase) - imag * np.sin(phase),
                                        real * np.sin(phase) + imag * np.cos(phase)), axis=-1)
                if config.noise_std:
                    payload = payload + self._noise_rng.normal(0, config.noise_std, payload.shape)
                limit = np.finfo(packet.payload.dtype).max
                payload = np.clip(np.nan_to_num(payload, nan=0., posinf=limit, neginf=-limit), -limit, limit).astype(packet.payload.dtype)
            if not (lost or congested):
                disturbed = replace(packet, payload=payload)
                self._queue.append((tick + config.latency_ticks + jitter, disturbed))
        self._bytes_by_tick[tick] = next_budget
        self._sender_ticks[sender_id] = tick
        self._last_send_tick = tick

    def deliver(self, timestep, *, receiver_id=None):
        timestep = _integer(timestep, 'timestep')
        if timestep < self._last_receive_tick:
            raise ValueError('delivery timestep moved backwards')
        if receiver_id is not None:
            receiver_id = _integer(receiver_id, 'receiver_id')
            if receiver_id not in self._regions:
                raise ValueError('unknown recipient')
        ready, pending = [], []
        for due, packet in self._queue:
            if due <= timestep and (receiver_id is None or packet.receiver_id == receiver_id):
                ready.append(packet)
            else:
                pending.append((due, packet))
        self._queue = pending
        self._last_receive_tick = timestep
        return tuple(sorted(ready, key=lambda p: (p.receiver_id, p.sender_id, p.timestep)))
