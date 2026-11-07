"""Authoritative topology must snapshot and validate every constructor input."""

import dataclasses

import numpy as np
import pytest

from aether.config import ModelConfig
from aether.neurofabric.topology import Topology


ARRAY_FIELDS = ('scalar_sources', 'scalar_targets', 'region_sources',
                'region_targets', 'neuron_regions')


def arrays():
    return dict(scalar_sources=np.array([0, 1, 2, 3]),
                scalar_targets=np.array([1, 2, 3, 0]),
                region_sources=np.array([0, 1]),
                region_targets=np.array([1, 0]),
                neuron_regions=np.array([0, 0, 1, 1]))


@pytest.mark.parametrize('name', ARRAY_FIELDS)
def test_topology_owns_immutable_snapshot_and_metadata_views(name):
    source = arrays()
    topology = Topology(**source)
    expected = source[name].copy()
    source[name].fill(100)
    np.testing.assert_array_equal(getattr(topology, name), expected)
    with pytest.raises(ValueError):
        getattr(topology, name)[0] = 7
    with pytest.raises(ValueError):
        getattr(topology, name).setflags(write=True)
    exposed = getattr(topology, name)
    exposed.shape = (1, -1)
    exposed.dtype = np.uint8
    assert getattr(topology, name).shape == expected.shape
    assert getattr(topology, name).dtype == np.int64
    np.testing.assert_array_equal(getattr(topology, name), expected)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(topology, name, expected)


@pytest.mark.parametrize('changes', [
    {'scalar_sources': np.array([[0, 1, 2, 3]])},
    {'scalar_sources': np.array([0., 1., 2., 3.])},
    {'scalar_sources': np.array([True, True, False, False])},
    {'scalar_sources': np.array([0, 1, 2])},
    {'scalar_sources': np.array([-1, 1, 2, 3])},
    {'scalar_targets': np.array([4, 2, 3, 0])},
    {'scalar_sources': np.array([0, 0, 2, 3]),
     'scalar_targets': np.array([1, 1, 3, 0])},
    {'scalar_targets': np.array([0, 2, 3, 0])},
    {'region_sources': np.array([0])},
    {'region_sources': np.array([-1, 1])},
    {'region_targets': np.array([2, 0])},
    {'region_sources': np.array([0, 0]), 'region_targets': np.array([1, 1])},
    {'region_targets': np.array([0, 0])},
    {'neuron_regions': np.array([], dtype=np.int64)},
    {'neuron_regions': np.array([[0, 0, 1, 1]])},
    {'neuron_regions': np.array([0., 0., 1., 1.])},
    {'neuron_regions': np.array([0, 0, 2, 2])},
    {'neuron_regions': np.array([1, 1, 2, 2])},
    {'neuron_regions': np.array([-1, 0, 1, 1])},
    {'scalar_sources': np.array([2**64 - 1, 1, 2, 3], dtype=np.uint64)},
    {'version': ''}, {'version': 1},
])
def test_constructor_rejects_malformed_or_unauthoritative_topology(changes):
    fields = arrays()
    fields.update(changes)
    with pytest.raises((TypeError, ValueError)):
        Topology(**fields)


def test_constructor_supports_nonempty_single_region_with_no_message_links():
    topology = Topology([0, 1], [1, 0], [], [], [0, 0])
    np.testing.assert_array_equal(topology.region_adjacency(), [[0.]])
    assert topology.region_sources.shape == (0,)
    assert topology.region_sources.dtype == np.int64


def test_seeded_builder_remains_reproducible_and_adjacency_is_derived():
    config = ModelConfig(neuron_count=16, scalar_edges=48, region_count=4,
                         region_message_edges=8, state_dimension=6,
                         mode_count=3, complex_dimensions=2)
    a, b = Topology.build(config, 51), Topology.build(config, 51)
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    adjacency = a.region_adjacency()
    assert adjacency.sum() == 8
    np.testing.assert_array_equal(adjacency[a.region_sources, a.region_targets], 1)
    adjacency.fill(9)
    assert a.region_adjacency().sum() == 8
