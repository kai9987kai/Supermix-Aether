import pytest

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
from aether.kernel.conformance import check_transport_conformance


@pytest.mark.parametrize('protocol', ['scalar_matched', 'vector', 'attention', 'phase', 'modes', 'wave'])
def test_clean_tensor_messages_match_validated_packet_reference(protocol):
    config = ModelConfig(neuron_count=16, scalar_edges=32, region_count=4,
                         region_message_edges=8, state_dimension=4,
                         mode_count=2, complex_dimensions=2)
    model = TissueModel(config, protocol, 7)
    result = check_transport_conformance(model)
    assert result['valid']
    assert result['payload_bytes_per_tick'] == 128
    assert result['max_absolute_error'] < 1e-6
