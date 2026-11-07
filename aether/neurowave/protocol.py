"""Immutable wire packets. Headers and queue memory are not payload bytes."""

from dataclasses import dataclass
import numbers

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


@dataclass(frozen=True, eq=False)
class NeuralPacket:
    """Snapshot a finite real-pair payload of shape (modes, complex_dim, 2).

    Payload dtype must match precision_bits (16, 32 or 64); callers explicitly
    quantize before construction. The copied array is backed by immutable bytes,
    so neither the original array nor setflags can modify the packet contents.
    Wire bytes are C-order little-endian IEEE floating point, without headers.
    """

    sender_id: int
    receiver_id: int
    timestep: int
    mode_count: int
    payload: np.ndarray
    precision_bits: int
    topology_version: int

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        # ndarray's read-only flag protects its storage, but shape and dtype
        # metadata remain assignable. A fresh view keeps those local to callers.
        return value.view() if name == 'payload' else value

    def __post_init__(self):
        for name in ('sender_id', 'receiver_id', 'timestep', 'topology_version'):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, 'mode_count', _integer(self.mode_count, 'mode_count', 1))
        precision = _integer(self.precision_bits, 'precision_bits', 1)
        if precision not in (16, 32, 64):
            raise ValueError('precision_bits must be 16, 32 or 64')
        object.__setattr__(self, 'precision_bits', precision)
        payload = np.asarray(self.payload)
        if payload.dtype.kind != 'f' or payload.dtype.itemsize * 8 != precision:
            raise ValueError('payload dtype must match precision_bits')
        if payload.ndim != 3 or payload.shape[0] != self.mode_count or payload.shape[1] < 1 or payload.shape[2] != 2:
            raise ValueError('payload must have shape (mode_count, positive complex_dim, 2)')
        if not np.isfinite(payload).all():
            raise ValueError('payload must contain only finite values')
        dtype = np.dtype(f'<f{precision // 8}')
        raw = payload.astype(dtype, copy=False).tobytes(order='C')
        immutable = np.frombuffer(raw, dtype=dtype).reshape(payload.shape)
        object.__setattr__(self, 'payload', immutable)

    @property
    def wire_bytes(self):
        """Little-endian payload bytes; excludes all metadata."""
        return self.payload.tobytes(order='C')

    @property
    def payload_bytes(self):
        return self.payload.nbytes
