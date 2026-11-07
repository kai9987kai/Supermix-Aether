"""Sparse differentiable scalar dynamics plus bounded regional communication.

All auxiliary arms instantiate identical learned tensors and FP16 wire shapes.
Only the routing calculation changes. Zero gain preserves the scalar calculation
exactly, while its derivative can open the auxiliary route during training.
Observation input reaches region 0 only; policy readout uses the final region.
The auxiliary update follows F_base + gain * clip(message, -1.0, 1.0).
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn

from aether.config import ModelConfig
from aether.neurofabric.topology import Topology
from aether.neurowave.graph_modes import graph_modes

PROTOCOLS = ('scalar', 'scalar_matched', 'vector', 'attention', 'phase', 'modes', 'wave')
INTERVENTIONS = ('none', 'zero_gain', 'erase_packets', 'shuffle_phase', 'random_phase')


def quantize_fp16(values):
    """IEEE FP16 forward quantization with identity straight-through gradient."""
    rounded = values.to(torch.float16).to(values.dtype)
    return values + (rounded - values).detach()


class TissueModel(nn.Module):
    def __init__(self, config: ModelConfig, protocol: str = 'wave', seed: int = 0):
        super().__init__()
        if protocol not in PROTOCOLS:
            raise ValueError(f'unknown protocol {protocol!r}')
        self.config, self.protocol, self.seed = config, protocol, int(seed)
        self.topology = Topology.build(config, seed)
        c = config
        for name in ('scalar_sources', 'scalar_targets', 'region_sources',
                     'region_targets', 'neuron_regions'):
            self.register_buffer(name, torch.tensor(getattr(self.topology, name).copy()))
        scalar_degree = np.bincount(self.topology.scalar_targets, minlength=c.neuron_count)
        self.register_buffer('scalar_degree', torch.tensor(np.maximum(scalar_degree, 1), dtype=torch.float32))
        region_degree = np.bincount(self.topology.region_targets, minlength=c.region_count)
        self.register_buffer('region_degree', torch.tensor(np.maximum(region_degree, 1), dtype=torch.float32))
        pool = torch.zeros(c.region_count, c.neuron_count)
        pool[self.neuron_regions, torch.arange(c.neuron_count)] = 1
        pool /= pool.sum(1, keepdim=True)
        self.register_buffer('pooling', pool)
        eigenvalues, carriers = graph_modes(self.topology.region_adjacency(), c.mode_count)
        self.register_buffer('carrier_eigenvalues', torch.tensor(eigenvalues, dtype=torch.float32))
        self.register_buffer('carriers', torch.tensor(carriers, dtype=torch.float32))
        mask = torch.zeros(c.region_count, c.region_count, dtype=torch.bool)
        mask[self.region_targets, self.region_sources] = True
        self.register_buffer('attention_mask', mask)
        profile = torch.linspace(0.6, 1.4, c.neuron_count)
        profile *= self.neuron_regions == 0
        self.register_buffer('input_profile', profile.view(1, -1, 1))
        self.register_buffer('readout_neurons', self.neuron_regions == c.region_count - 1)
        # Forking prevents caller RNG side effects. Base initialization happens
        # before auxiliary initialization in every arm and is exactly paired.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input_encoder = nn.Linear(c.input_dimension, c.state_dimension)
            self.output_decoder = nn.Linear(c.state_dimension, c.action_count)
            self.local_transform = nn.Parameter(torch.eye(c.state_dimension) * 0.15)
            self.self_gain = nn.Parameter(torch.full((c.state_dimension,), 0.75))
            self.neuron_bias = nn.Parameter(torch.randn(c.neuron_count, c.state_dimension) * 0.03)
            self.edge_weights = nn.Parameter(torch.ones(c.scalar_edges) * 0.4)
            if protocol != 'scalar':
                width = c.mode_count * c.complex_dimensions * 2
                self.message_encoder = nn.Linear(c.state_dimension, width)
                self.message_decoder = nn.Linear(width, c.state_dimension, bias=False)
                self.routing_query = nn.Linear(c.state_dimension, c.state_dimension, bias=False)
                self.routing_key = nn.Linear(c.state_dimension, c.state_dimension, bias=False)
                self.phase_encoder = nn.Linear(c.state_dimension, c.complex_dimensions)
                self.mode_gains = nn.Parameter(torch.ones(c.mode_count))
                self.residual_gain = nn.Parameter(torch.tensor(float(c.initial_residual_gain)))

    def initial_state(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        return self.input_encoder.weight.new_zeros(batch_size, self.config.neuron_count,
                                                  self.config.state_dimension)

    def _local(self, state):
        c, batch = self.config, state.shape[0]
        indices = torch.stack((self.scalar_targets, self.scalar_sources))
        flat = state.permute(1, 0, 2).reshape(c.neuron_count, -1)
        # A documented local context opts in explicitly (PyTorch 2.11 otherwise
        # warns about its implicit global default). The caller's flag is restored.
        with torch.sparse.check_sparse_tensor_invariants(True):
            adjacency = torch.sparse_coo_tensor(
                indices, self.edge_weights / self.scalar_degree[self.scalar_targets],
                (c.neuron_count, c.neuron_count), check_invariants=True)
            aggregate = torch.sparse.mm(adjacency, flat)
        return aggregate.reshape(c.neuron_count, batch, c.state_dimension).permute(1, 0, 2)

    def _phase(self, pooled, intervention, disturbance_seed):
        phases = math.pi * torch.tanh(self.phase_encoder(pooled))
        if intervention in ('shuffle_phase', 'random_phase'):
            generator = torch.Generator(device=pooled.device).manual_seed(int(disturbance_seed))
            if intervention == 'shuffle_phase':
                order = torch.randperm(self.config.region_count, generator=generator, device=pooled.device)
                phases = phases[:, order]
            else:
                phases = torch.rand(phases.shape, generator=generator,
                                    device=pooled.device, dtype=pooled.dtype) * (2 * math.pi) - math.pi
        return phases

    def encoded_payload(self, state, intervention='none', disturbance_seed=0):
        """Return the delivered wire values [batch, link, mode, complex, 2].

        Values are FP16-rounded in the state's dtype to retain autograd. Casting
        the detached result to NumPy float16 produces packet-reference payloads.
        Scalar-only models have no auxiliary route and reject this operation.
        """
        if intervention not in INTERVENTIONS:
            raise ValueError(f'unknown intervention {intervention!r}')
        if self.protocol == 'scalar':
            raise ValueError('scalar protocol has no auxiliary payload')
        c = self.config
        if state.ndim != 3 or tuple(state.shape[1:]) != (c.neuron_count, c.state_dimension):
            raise ValueError('state must have shape [batch, neuron_count, state_dimension]')
        pooled = torch.matmul(self.pooling, state)
        encoded = self.message_encoder(pooled).reshape(
            state.shape[0], c.region_count, c.mode_count, c.complex_dimensions, 2)
        if self.protocol == 'scalar_matched':
            # Learned encoder scalar bottleneck; fixed dense wire shape.
            encoded = encoded.mean(dim=(2, 3, 4), keepdim=True).expand_as(encoded)
        if self.protocol in ('modes', 'wave'):
            encoded = encoded * (self.carriers * self.mode_gains).view(1, c.region_count, c.mode_count, 1, 1)
        if self.protocol in ('phase', 'wave'):
            phases = self._phase(pooled, intervention, disturbance_seed)
            cosine, sine = phases.cos().unsqueeze(2), phases.sin().unsqueeze(2)
            real, imag = encoded[..., 0], encoded[..., 1]
            encoded = torch.stack((real * cosine - imag * sine,
                                   real * sine + imag * cosine), dim=-1)
            # Mode 0 carries a unit-amplitude phase pilot. Its cost is inside the
            # fixed wire budget: 1/K of complex slots (25% with default K=4).
            pilot = torch.stack((phases.cos(), phases.sin()), dim=-1).unsqueeze(2)
            encoded = torch.cat((pilot, encoded[:, :, 1:]), dim=2)
        # One immutable-shape payload per directed link; FP16 rounding occurs
        # before delivery, matching the packet reference precision.
        packets = quantize_fp16(encoded[:, self.region_sources])
        if intervention == 'erase_packets':
            packets = packets * 0
        return packets

    def _messages(self, state, intervention, disturbance_seed, packet_transform=None):
        packets = self.encoded_payload(state, intervention, disturbance_seed)
        if packet_transform is not None:
            packets = packet_transform(packets)
        return self.receiver_decode(packets, state)

    def receiver_decode(self, packets, state):
        """Decode only wire values plus each receiver's own local state.

        Sender state, keys and phases are never read here. Interventions affect
        the encoded wire; receiver phase references remain locally computed.
        """
        c = self.config
        if self.protocol == 'scalar':
            raise ValueError('scalar protocol has no auxiliary payload')
        if tuple(packets.shape) != (state.shape[0], c.region_message_edges,
                                     c.mode_count, c.complex_dimensions, 2):
            raise ValueError('invalid packet tensor shape')
        pooled = torch.matmul(self.pooling, state)
        weights = (1 / self.region_degree[self.region_targets]).view(1, -1)
        if self.protocol == 'attention':
            query = self.routing_query(pooled)[:, self.region_targets]
            key = self.routing_key(self.message_decoder(packets.flatten(2)))
            scores = (query * key).sum(-1) / math.sqrt(c.state_dimension)
            weights = torch.zeros_like(scores)
            for region in range(c.region_count):
                incoming = self.region_targets == region
                if incoming.any():
                    weights[:, incoming] = scores[:, incoming].softmax(dim=1)
        if self.protocol in ('phase', 'wave'):
            receiver_phases = self._phase(pooled, 'none', 0)[:, self.region_targets]
            pilot = packets[:, :, 0]
            sender_phases = torch.atan2(pilot[..., 1], pilot[..., 0])
            # Bounded cosine phase coherence gates directed delivery. This is a
            # heuristic communication gate, not a biological phase claim.
            difference = sender_phases - receiver_phases
            coherence = (1 + difference.cos().mean(-1)) * 0.5
            weights = weights * coherence
        packets = packets * weights[..., None, None, None]
        delivered = state.new_zeros(state.shape[0], c.region_count,
                                    c.mode_count, c.complex_dimensions, 2)
        delivered.index_add_(1, self.region_targets, packets)
        if self.protocol in ('modes', 'wave'):
            delivered = delivered * self.carriers.view(1, c.region_count, c.mode_count, 1, 1)
        decoded = self.message_decoder(delivered.flatten(2))
        return decoded[:, self.neuron_regions]

    def step(self, observation, state, intervention='none', disturbance_seed=0,
             packet_transform=None):
        if intervention not in INTERVENTIONS:
            raise ValueError(f'unknown intervention {intervention!r}')
        c = self.config
        if observation.ndim != 2 or observation.shape[1] != c.input_dimension:
            raise ValueError('observation must have shape [batch, input_dimension]')
        if tuple(state.shape) != (observation.shape[0], c.neuron_count, c.state_dimension):
            raise ValueError('state must have shape [batch, neuron_count, state_dimension]')
        drive = (state * self.self_gain + self._local(state) @ self.local_transform
                 + self.input_encoder(observation).unsqueeze(1) * self.input_profile + self.neuron_bias)
        next_state = torch.tanh(drive)
        if self.protocol != 'scalar':
            enabled = intervention not in ('zero_gain', 'erase_packets')
            if enabled or packet_transform is not None:
                message = self._messages(state, intervention, disturbance_seed, packet_transform)
                if enabled:
                    next_state = next_state + self.residual_gain * message.clamp(-1.0, 1.0)
        logits = self.output_decoder(next_state[:, self.readout_neurons].mean(1))
        return logits, next_state

    def forward(self, observations, intervention='none', disturbance_seed=0):
        if observations.ndim != 3 or observations.shape[-1] != self.config.input_dimension or observations.shape[1] < 1:
            raise ValueError('observations must have shape [batch, positive time, input_dimension]')
        state = self.initial_state(observations.shape[0])
        logits = []
        for tick in range(observations.shape[1]):
            output, state = self.step(observations[:, tick], state, intervention,
                                      disturbance_seed=int(disturbance_seed) + tick)
            logits.append(output)
        return torch.stack(logits, dim=1)

    def state_digest(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Restore one authoritative graph and reject inconsistent derivations.

        Structural validation occurs before tensors are copied, so a checkpoint
        cannot silently pair loaded routing buffers with a stale Python topology.
        Tensor shape, dtype and layout must match; implicit dtype casts are
        rejected. Failed loads preserve tensor values and parameter identities.
        """
        c = self.config
        current = self.state_dict()
        supplied_keys, current_keys = set(state_dict), set(current)
        if strict and supplied_keys != current_keys:
            missing = sorted(current_keys - supplied_keys)
            unexpected = sorted(supplied_keys - current_keys)
            raise RuntimeError(f'checkpoint key mismatch: missing={missing}, unexpected={unexpected}')
        for name, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'checkpoint value {name!r} must be a tensor')
            if name in current:
                expected_tensor = current[name]
                if value.shape != expected_tensor.shape:
                    raise RuntimeError(f'checkpoint shape mismatch for {name}')
                if value.dtype != expected_tensor.dtype or value.layout != expected_tensor.layout:
                    raise RuntimeError(f'checkpoint dtype/layout mismatch for {name}')
        candidate = dict(current)
        candidate.update(state_dict)
        graph_names = ('scalar_sources', 'scalar_targets', 'region_sources',
                       'region_targets', 'neuron_regions')
        arrays = [candidate[name].detach().cpu().numpy().copy() for name in graph_names]
        topology = Topology(*arrays)
        if (len(topology.scalar_sources) != c.scalar_edges
                or len(topology.region_sources) != c.region_message_edges
                or len(topology.neuron_regions) != c.neuron_count
                or int(topology.neuron_regions.max()) + 1 != c.region_count):
            raise ValueError('checkpoint topology does not match model config')
        sources, targets = topology.scalar_sources, topology.scalar_targets
        r_sources, r_targets = topology.region_sources, topology.region_targets
        regions = torch.tensor(topology.neuron_regions.copy())
        pooling = torch.zeros(c.region_count, c.neuron_count)
        pooling[regions, torch.arange(c.neuron_count)] = 1
        pooling /= pooling.sum(1, keepdim=True)
        mask = torch.zeros(c.region_count, c.region_count, dtype=torch.bool)
        mask[torch.tensor(r_targets.copy()), torch.tensor(r_sources.copy())] = True
        profile = torch.linspace(0.6, 1.4, c.neuron_count) * (regions == 0)
        eigenvalues, carriers = graph_modes(topology.region_adjacency(), c.mode_count)
        expected = {
            'scalar_degree': torch.tensor(np.maximum(np.bincount(targets, minlength=c.neuron_count), 1)),
            'region_degree': torch.tensor(np.maximum(np.bincount(r_targets, minlength=c.region_count), 1)),
            'pooling': pooling,
            'attention_mask': mask,
            'input_profile': profile.view(1, -1, 1),
            'readout_neurons': regions == c.region_count - 1,
            'carrier_eigenvalues': torch.tensor(eigenvalues, dtype=torch.float32),
            'carriers': torch.tensor(carriers, dtype=torch.float32),
        }
        for name, desired in expected.items():
            actual = candidate[name].detach().cpu()
            desired = desired.to(actual.dtype)
            if actual.shape != desired.shape or not torch.allclose(actual, desired, rtol=1e-6, atol=1e-7):
                raise ValueError(f'checkpoint has inconsistent derived buffer {name}')
        # Preflight covers ordinary PyTorch errors. Retain original objects and
        # values as a transactional fallback for load hooks or copy failures,
        # including assign=True, which can replace Parameter registrations.
        originals = [(name, value, value.detach().clone()) for name, value in
                     list(self.named_parameters(remove_duplicate=False))
                     + list(self.named_buffers(remove_duplicate=False))]
        try:
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        except Exception:
            with torch.no_grad():
                for name, original, snapshot in originals:
                    path, _, leaf = name.rpartition('.')
                    setattr(self.get_submodule(path), leaf, original)
                    original.copy_(snapshot)
            raise
        self.topology = topology
        return result

    def resource_metrics(self):
        c = self.config
        from .profiler import count_step_flops
        return {
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in self.parameters()),
            'persistent_buffer_bytes': sum(b.numel() * b.element_size() for b in self.buffers()),
            'step_flops': count_step_flops(self),
            'payload_bytes_per_tick': 0 if self.protocol == 'scalar' else
                c.region_message_edges * c.mode_count * c.complex_dimensions * 2 * 2,
            'payload_accounting': 'FP16 payload only; excludes headers, queues, activations and scalar route',
            'phase_pilot_fraction': 1 / c.mode_count if self.protocol in ('phase', 'wave') else 0.0,
        }

    def resource_summary(self):
        return self.resource_metrics()

    def profile(self, warmup_steps: int = 5, benchmark_steps: int = 25, batch_size: int = 1):
        from .profiler import profile_model
        return profile_model(self, warmup_steps=warmup_steps,
                             benchmark_steps=benchmark_steps,
                             batch_size=batch_size)
