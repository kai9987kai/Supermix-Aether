"""Serializable bounds for the fixed-topology first milestone."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    neuron_count: int = 512
    scalar_edges: int = 4096
    region_count: int = 16
    region_message_edges: int = 96
    state_dimension: int = 32
    mode_count: int = 4
    complex_dimensions: int = 8
    input_dimension: int = 12
    action_count: int = 7
    initial_residual_gain: float = 0.0

    def __post_init__(self):
        names = ('neuron_count', 'region_count', 'state_dimension', 'mode_count',
                 'complex_dimensions', 'input_dimension', 'action_count')
        for name in names:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.region_count > self.neuron_count:
            raise ValueError('region_count cannot exceed neuron_count')
        if self.mode_count > self.region_count:
            raise ValueError('mode_count cannot exceed region_count')
        for name, count in (('scalar_edges', self.neuron_count),
                            ('region_message_edges', self.region_count)):
            edges = getattr(self, name)
            if type(edges) is not int or not 0 <= edges <= count * (count - 1):
                raise ValueError(f'{name} exceeds the directed simple graph capacity')
        if not math.isfinite(self.initial_residual_gain):
            raise ValueError('initial_residual_gain must be finite')

    def to_dict(self):
        return asdict(self)

