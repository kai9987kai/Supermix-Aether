"""Seeded interventions on actual fixed-shape FP16 message tensors.

Bandwidth admits whole link packets; loss then erases admitted packets. Noise
affects surviving packets only and is rounded to FP16 before receiver decode.
Latency is a fixed FIFO delay of at most four model ticks, with zero warmup.
Counters count payload values only, excluding headers and scalar communication.
"""
from collections import deque
import math
from numbers import Real

import torch


def validate_channel_settings(*, loss_probability=0.0, noise_std=0.0,
                              latency_ticks=0, bandwidth_fraction=1.0):
    for name, value in [('loss_probability', loss_probability), ('noise_std', noise_std),
                        ('bandwidth_fraction', bandwidth_fraction)]:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be a finite number in [0, 1]')
    if type(latency_ticks) is not int or not 0 <= latency_ticks <= 4:
        raise ValueError('latency_ticks must be an integer in [0, 4]')


class PacketChannel:
    def __init__(self, *, seed=0, loss_probability=0.0, noise_std=0.0,
                 latency_ticks=0, bandwidth_fraction=1.0):
        validate_channel_settings(loss_probability=loss_probability, noise_std=noise_std,
                                  latency_ticks=latency_ticks, bandwidth_fraction=bandwidth_fraction)
        self.loss_probability, self.noise_std = float(loss_probability), float(noise_std)
        self.latency_ticks, self.bandwidth_fraction = latency_ticks, float(bandwidth_fraction)
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed) % (2**63 - 1))
        self.queue = deque(maxlen=4)
        self.attempted_payload_bytes = self.admitted_payload_bytes = 0
        self.surviving_payload_bytes = self.delivered_payload_bytes = 0
        self.last_transmitted = self.last_delivered = None

    def __call__(self, packets):
        if packets.ndim != 5 or packets.device.type != 'cpu':
            raise ValueError('live packets must be CPU [batch, link, mode, complex, 2] tensors')
        batch, links = packets.shape[:2]
        bytes_per_link = math.prod(packets.shape[2:]) * 2
        self.attempted_payload_bytes = packets.numel() * 2
        admitted_count = int(links * self.bandwidth_fraction)
        self.admitted_payload_bytes = batch * admitted_count * bytes_per_link
        clean = self.bandwidth_fraction == 1 and self.loss_probability == 0 and self.noise_std == 0
        if clean:
            transmitted = packets
            surviving_bytes = self.admitted_payload_bytes
        else:
            admitted = torch.ones((batch, links), dtype=torch.bool)
            if admitted_count < links:
                admitted.zero_()
                for row in range(batch):
                    order = torch.randperm(links, generator=self.generator)
                    admitted[row, order[:admitted_count]] = True
            if self.loss_probability:
                admitted &= torch.rand((batch, links), generator=self.generator) >= self.loss_probability
            values = packets
            if self.noise_std:
                noise = torch.randn(packets.shape, generator=self.generator, dtype=packets.dtype)
                values = (packets + self.noise_std * noise).half().to(packets.dtype)
            transmitted = values * admitted[..., None, None, None]
            surviving_bytes = int(admitted.sum()) * bytes_per_link
        self.last_transmitted = transmitted.detach().clone()
        self.surviving_payload_bytes = surviving_bytes
        if self.latency_ticks:
            if len(self.queue) >= self.latency_ticks:
                delivered, delivered_bytes = self.queue.popleft()
            else:
                delivered, delivered_bytes = torch.zeros_like(packets), 0
            self.queue.append((transmitted.detach().clone(), surviving_bytes))
        else:
            delivered, delivered_bytes = transmitted, surviving_bytes
        self.delivered_payload_bytes = delivered_bytes
        self.last_delivered = delivered.detach().clone()
        return delivered
