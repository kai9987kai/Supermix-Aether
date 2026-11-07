"""Immutable topology and SynapseForge structural plasticity."""
from .topology import Topology
from .plasticity import (
    PlasticityReport,
    lesion_region_links,
    reconcile_model_topology,
    HebbianPlasticityEngine,
)

__all__ = [
    'Topology',
    'PlasticityReport',
    'lesion_region_links',
    'reconcile_model_topology',
    'HebbianPlasticityEngine',
]
