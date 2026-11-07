"""Pure paired live execution; scheduling and thread serialization belong to HTTP.

Snapshots contain cached measurements from the latest actual inference in each
branch. Goal coordinates are spectator metadata; the policy receives only the
environment's observation, which exposes its goal cue at tick zero.
"""
from collections import deque
from copy import deepcopy

import numpy as np
import torch

from aether.environments.navigation import NavigationWorld
from aether.kernel.runtime import INTERVENTIONS, TissueModel
from aether.live.channel import PacketChannel, validate_channel_settings
from aether.live.models import clone_model


class LiveSession:
    def __init__(self, model, *, seed=2077, delay=3, horizon=9, provenance=None):
        self._validate_seed(seed)
        self._validate_delay(delay)
        if type(horizon) is not int or horizon != delay + 6:
            raise ValueError('live horizon must equal delay + 6')
        self.model = self._copy_model(model)
        self._provenance = deepcopy(provenance or {'kind': 'fresh'})
        self._settings = dict(intervention='none', loss_probability=0.0, noise_std=0.0,
                              latency_ticks=0, bandwidth_fraction=1.0, delay=delay,
                              task_family='straight')
        self.seed, self.sequence, self.episode = seed, 0, 0
        self.provenance_epoch = 0
        self._trace = deque(maxlen=512)
        self._trace_total = 0
        self._describe_model()
        self._restart()

    @staticmethod
    def _validate_seed(seed):
        if type(seed) is not int or not 0 <= seed < 2**63 - 1:
            raise ValueError('seed must be an integer in [0, 2**63 - 1)')

    @staticmethod
    def _validate_delay(delay):
        if type(delay) is not int or not 1 <= delay <= 12:
            raise ValueError('delay must be an integer in [1, 12]')

    @staticmethod
    def _copy_model(model):
        if not isinstance(model, TissueModel):
            raise ValueError('live execution requires a TissueModel')
        if model.config.input_dimension != 12 or model.config.action_count != 7:
            raise ValueError('navigation requires 12 observation values and 7 actions')
        if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
            raise ValueError('live execution requires CPU weights')
        return clone_model(model).eval()

    @property
    def settings(self):
        return deepcopy(self._settings)

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    @property
    def delay(self):
        return self._settings['delay']

    @property
    def horizon(self):
        return self.delay + 6

    def _describe_model(self):
        c = self.model.config
        from aether.kernel.profiler import count_step_flops
        flops = count_step_flops(self.model)
        self._model_info = dict(protocol=self.model.protocol, neurons=c.neuron_count,
            regions=c.region_count, modes=c.mode_count, digest=self.model.state_digest(),
            residual_gain=float(self.model.residual_gain.detach()) if self.model.protocol != 'scalar' else 0.0,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            step_flops=flops)
        self._links = list(map(list, zip(self.model.region_sources.tolist(), self.model.region_targets.tolist())))

    def _restart(self):
        task_family = self._settings.get('task_family', 'straight')
        from aether.environments.stress import create_world
        self.candidate_world = create_world(task_family, seed=self.seed, delay=self.delay, horizon=self.horizon)
        self.control_world = self.candidate_world.clone()
        self.candidate_state = self.model.initial_state(1)
        self.control_state = self.candidate_state.clone()
        channel_settings = {key: value for key, value in self._settings.items() if key not in ('intervention', 'delay', 'task_family')}
        self.candidate_channel = PacketChannel(seed=self.seed, **channel_settings)
        self.control_channel = PacketChannel(seed=self.seed)
        self._history = deque(maxlen=512)
        self._effective_differences = self._waiting_differences = 0
        self._branches = {name: self._initial_branch() for name in ('candidate', 'control')}
        self._cache_frame()

    def _initial_branch(self):
        return dict(position=[0, 0, 0], trajectory=[[0, 0, 0]],
            goal=self.candidate_world.goal.tolist(), success=False, done=False,
            **{'return': 0.0}, action=None, probabilities=[0.0] * 7,
            regions=[dict(id=i, activity=0.0) for i in range(self.model.config.region_count)],
            mode_energy=[0.0] * self.model.config.mode_count,
            links=[dict(source=s, target=t, energy=0.0) for s, t in self._links],
            payload_bytes=0, attempted_payload_bytes=0, admitted_payload_bytes=0,
            surviving_payload_bytes=0, delivered_payload_bytes=0,
            telemetry_tick=None)

    def _cache_frame(self):
        candidate, control = self._branches['candidate'], self._branches['control']
        self._frame = dict(schema='aether.live.frame.v1', sequence=self.sequence,
            episode=self.episode, seed=self.seed,
            tick=max(self.candidate_world.timestep, self.control_world.timestep),
            done=self.candidate_world.done and self.control_world.done,
            settings=deepcopy(self._settings), provenance=deepcopy(self._provenance),
            provenance_epoch=self.provenance_epoch, model=deepcopy(self._model_info),
            topology={'links': deepcopy(self._links)},
            candidate=deepcopy(candidate), control=deepcopy(control),
            comparison=dict(effective_action_differences=self._effective_differences,
                waiting_action_differences=self._waiting_differences,
                position_distance=float(np.linalg.norm(self.candidate_world.position - self.control_world.position)),
                return_delta=candidate['return'] - control['return']),
            history=list(self._history),
            payload_accounting='FP16 payload only: attempted before bandwidth, admitted before loss, surviving after loss, delivered after latency; excludes headers/scalar route. Link/mode energy is mean square of receiver-input packets.',
            observation_boundary='Goal is spectator metadata; policy receives initial cue only through NavigationWorld.observe().')

    def snapshot(self):
        return deepcopy(self._frame)

    def reset(self, seed=None):
        chosen = self.seed if seed is None else seed
        self._validate_seed(chosen)
        self.seed = chosen
        self.episode += 1
        self.sequence += 1
        self._restart()
        return self.snapshot()

    def configure(self, **settings):
        if set(settings) - set(self._settings):
            raise ValueError('unknown live setting')
        candidate = dict(self._settings, **settings)
        self._validate_delay(candidate['delay'])
        if candidate['intervention'] not in INTERVENTIONS:
            raise ValueError('unknown intervention')
        if candidate.get('task_family') not in ('straight', 'distraction', 'waypoint'):
            raise ValueError('unknown task_family')
        validate_channel_settings(**{key: value for key, value in candidate.items()
                                     if key not in ('delay', 'intervention', 'task_family')})
        if candidate != self._settings:
            self._settings = candidate
            return self.reset()
        return self.snapshot()

    def lesion_links(self, severed_pairs):
        from aether.neurofabric.plasticity import lesion_region_links, reconcile_model_topology
        new_top = lesion_region_links(self.model.topology, severed_pairs)
        reconcile_model_topology(self.model, new_top)
        self._describe_model()
        return self.reset()

    def rewire_hebbian(self):
        from aether.neurofabric.plasticity import HebbianPlasticityEngine
        engine = HebbianPlasticityEngine(pruning_percentile=50.0, rewiring_budget=2)
        # Gather regional activity history if available, else synthetic baseline
        activity = []
        for step in self._history:
            cand = step.get('candidate_regions', [])
            if cand:
                activity.append([r['activity'] for r in cand])
        if len(activity) < 2:
            activity = np.random.default_rng(self.seed).standard_normal((10, self.model.config.region_count))
        else:
            activity = np.array(activity)
        engine.adapt_region_links(self.model, activity)
        self._describe_model()
        return self.reset()

    def replace_model(self, model, provenance=None):
        replacement = self._copy_model(model)
        copied_provenance = deepcopy(provenance or {'kind': 'session-adapted'})
        self.model, self._provenance = replacement, copied_provenance
        self.provenance_epoch += 1
        self._trace.clear()
        self._trace_total = 0
        self._describe_model()
        return self.reset()

    def _advance(self, name):
        world = getattr(self, name + '_world')
        if world.done:
            return None, 0.0
        state, channel = getattr(self, name + '_state'), getattr(self, name + '_channel')
        intervention = self._settings['intervention'] if name == 'candidate' else 'none'
        observation = torch.from_numpy(world.observe()).unsqueeze(0)
        logits, next_state = self.model.step(observation, state, intervention=intervention,
            disturbance_seed=(self.seed + world.timestep) % (2**63 - 1), packet_transform=channel)
        probabilities = logits.softmax(-1)[0]
        action = int(probabilities.argmax())
        _, reward, _ = world.step(action)
        setattr(self, name + '_state', next_state)
        branch = self._branches[name]
        branch.update(position=world.position.tolist(), success=world.success, done=world.done,
                      action=action, probabilities=probabilities.tolist(), telemetry_tick=world.timestep)
        branch['return'] += reward
        branch['trajectory'].append(world.position.tolist())
        branch['regions'] = [dict(id=region, activity=float(next_state[:, self.model.neuron_regions == region].square().mean().sqrt()))
                             for region in range(self.model.config.region_count)]
        packets = channel.last_delivered
        if packets is not None and packets.numel():
            link_energy = packets.square().mean(dim=(0, 2, 3, 4)).tolist()
            branch['mode_energy'] = packets.square().mean(dim=(0, 1, 3, 4)).tolist()
            branch['links'] = [dict(source=s, target=t, energy=energy)
                               for (s, t), energy in zip(self._links, link_energy)]
        for field in ('attempted_payload_bytes', 'admitted_payload_bytes', 'surviving_payload_bytes', 'delivered_payload_bytes'):
            branch[field] = getattr(channel, field)
        branch['payload_bytes'] = channel.admitted_payload_bytes
        return action, reward

    def step(self):
        if self._frame['done']:
            self.reset(seed=(self.seed + 1) % (2**63 - 1))
        movement_enabled = max(self.candidate_world.timestep, self.control_world.timestep) >= self.delay
        with torch.inference_mode():
            action, reward = self._advance('candidate')
            control_action, control_reward = self._advance('control')
        if action is not None and control_action is not None and action != control_action:
            if movement_enabled:
                self._effective_differences += 1
            else:
                self._waiting_differences += 1
        self.sequence += 1
        self._history.append(dict(tick=max(self.candidate_world.timestep, self.control_world.timestep),
            action=action, control_action=control_action, movement_enabled=movement_enabled,
            reward=reward, control_reward=control_reward,
            paired_actions_comparable=action is not None and control_action is not None))
        self._cache_frame()
        # Export stores compact actual events, not redundant neural-state frames.
        self._trace.append(dict(sequence=self.sequence, episode=self.episode, seed=self.seed,
            settings=self.settings, provenance_epoch=self.provenance_epoch,
            model_digest=self._model_info['digest'], **self._history[-1],
            position=self.candidate_world.position.tolist(), control_position=self.control_world.position.tolist()))
        self._trace_total += 1
        return self.snapshot()

    def export(self):
        return deepcopy(dict(schema='aether.live.trace.v1', exploratory=True,
            full_neural_state_replay=False, truncated=self._trace_total > len(self._trace),
            observed_steps=self._trace_total, retained_steps=len(self._trace),
            limitations='Bounded exploratory executed-action trace; not confirmatory evidence or a full neural-state replay.',
            frame=self._frame, frames=list(self._trace)))
