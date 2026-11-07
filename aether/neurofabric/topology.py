"""One authoritative, reproducible directed topology; growth is disabled."""
from dataclasses import dataclass
import numpy as np

from aether.config import ModelConfig


_ARRAY_FIELDS = frozenset(('scalar_sources', 'scalar_targets', 'region_sources',
                           'region_targets', 'neuron_regions'))


def _immutable(values):
    array = np.asarray(values, dtype=np.int64)
    # A bytes-backed array cannot be made writable by a caller.
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _edges(count, size, rng):
    # Start with a ring where budget permits, then sample without replacement.
    # Enumerating possible edges is constructor-only, never a runtime N x N tensor.
    ring = [(i, (i + 1) % count) for i in range(count)] if count > 1 else []
    if size <= len(ring):
        chosen = ring[:size]
    else:
        used = set(ring)
        candidates = [(a, b) for a in range(count) for b in range(count)
                      if a != b and (a, b) not in used]
        indexes = rng.choice(len(candidates), size - len(ring), replace=False)
        chosen = ring + [candidates[i] for i in indexes]
    chosen.sort(key=lambda pair: (pair[1], pair[0]))
    if not chosen:
        return _immutable([]), _immutable([])
    return _immutable([a for a, _ in chosen]), _immutable([b for _, b in chosen])


@dataclass(frozen=True)
class Topology:
    """Validated immutable snapshot of scalar and regional directed graphs.

    The five arrays must be integer vectors. ``neuron_regions`` supplies one
    region ID for each neuron; IDs must cover 0..region_count-1 with no empty
    regions. Edges are unique, directed, nonself and in range. Construct this
    class directly from loaded runtime buffer arrays to restore the authoritative
    snapshot after checkpoint loading. Public ndarray accesses return fresh
    views so caller changes to shape/dtype metadata cannot alter the topology.
    """

    scalar_sources: np.ndarray
    scalar_targets: np.ndarray
    region_sources: np.ndarray
    region_targets: np.ndarray
    neuron_regions: np.ndarray
    version: str = 'fixed-v1'

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return value.view() if name in _ARRAY_FIELDS and isinstance(value, np.ndarray) else value

    def __eq__(self, other):
        if not isinstance(other, Topology):
            return False
        return (self.version == other.version and
                all(np.array_equal(getattr(self, k), getattr(other, k)) for k in _ARRAY_FIELDS))

    def __post_init__(self):
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError('topology version must be a nonempty string')
        arrays = {}
        for name in _ARRAY_FIELDS:
            raw = np.asarray(object.__getattribute__(self, name))
            if raw.ndim != 1:
                raise ValueError(f'{name} must be a one-dimensional integer vector')
            # Empty edge sequences have no values to coerce, including Python
            # [] whose inferred dtype is float64. Nonempty inputs stay strict.
            if raw.size and raw.dtype.kind not in 'iu':
                raise TypeError(f'{name} must contain integers')
            if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
                raise ValueError(f'{name} contains an invalid nonnegative ID')
            arrays[name] = _immutable(raw)
        regions = arrays['neuron_regions']
        if not regions.size:
            raise ValueError('topology must contain at least one neuron')
        unique_regions = np.unique(regions)
        if not np.array_equal(unique_regions, np.arange(len(unique_regions))):
            raise ValueError('region IDs must be contiguous from zero and nonempty')
        for prefix, count in (('scalar', len(regions)), ('region', len(unique_regions))):
            sources, targets = arrays[f'{prefix}_sources'], arrays[f'{prefix}_targets']
            if sources.shape != targets.shape:
                raise ValueError(f'{prefix} source/target counts must match')
            if np.any(sources >= count) or np.any(targets >= count):
                raise ValueError(f'{prefix} endpoint out of range')
            if np.any(sources == targets):
                raise ValueError(f'{prefix} self edges are forbidden')
            if len(set(zip(sources.tolist(), targets.tolist()))) != len(sources):
                raise ValueError(f'{prefix} edges must be unique')
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @classmethod
    def build(cls, config: ModelConfig, seed: int = 0):
        local_rng, region_rng = [np.random.default_rng(s) for s in
                                 np.random.SeedSequence(seed).spawn(2)]
        sources, targets = _edges(config.neuron_count, config.scalar_edges, local_rng)
        region_sources, region_targets = _edges(
            config.region_count, config.region_message_edges, region_rng)
        regions = _immutable(np.arange(config.neuron_count) * config.region_count
                             // config.neuron_count)
        return cls(sources, targets, region_sources, region_targets, regions)

    def region_adjacency(self):
        adjacency = np.zeros((int(self.neuron_regions.max()) + 1,) * 2)
        adjacency[self.region_sources, self.region_targets] = 1.0
        return adjacency

