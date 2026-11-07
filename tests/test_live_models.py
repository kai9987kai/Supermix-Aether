import pytest
import torch

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
from aether.live.models import clone_model


def test_clone_preserves_weights_and_authoritative_topology_without_aliasing():
    config = ModelConfig(neuron_count=8, scalar_edges=16, region_count=2,
                         region_message_edges=2, state_dimension=4,
                         mode_count=2, complex_dimensions=2)
    original = TissueModel(config, 'wave', 7)
    clone = clone_model(original)
    assert clone.state_digest() == original.state_digest()
    assert clone.input_encoder.weight.data_ptr() != original.input_encoder.weight.data_ptr()
    with pytest.raises(ValueError):
        clone.topology.scalar_sources[0] = 4
    before = original.state_digest()
    with torch.no_grad():
        clone.input_encoder.weight.add_(1)
    assert original.state_digest() == before
