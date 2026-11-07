"""Pure paired live execution; scheduling and thread serialization belong to HTTP.

Snapshots contain cached measurements from the latest actual inference in each
branch. Goal coordinates are spectator metadata; the policy receives only the
environment's observation, which exposes its goal cue at tick zero.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict

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
        self._structural_report = None
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
        self._activity_history = deque(maxlen=512)
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

    @staticmethod
    def _task_info(world):
        info = dict(radius=world.radius, horizon=world.horizon,
                    stage=1, waypoints=[world.goal.tolist()])
        if hasattr(world, 'waypoint1'):
            info.update(stage=2 if world.visited_waypoint1 else 1,
                        waypoints=[world.waypoint1.tolist(), world.waypoint2.tolist()],
                        visited_waypoint1=world.visited_waypoint1)
        info['sensory_cue'] = world.observe()[:6].tolist()
        return info

    def _cache_frame(self):
        candidate, control = self._branches['candidate'], self._branches['control']
        for name, branch in self._branches.items():
            world = getattr(self, name + '_world')
            branch.update(goal=world.goal.tolist(), task=self._task_info(world))
        family = self._settings['task_family']
        boundary = {
            'straight': 'Policy receives the initial directional cue only at tick zero.',
            'distraction': 'Policy receives the true cue at tick zero and false sensory pulses during the waiting delay.',
            'waypoint': 'Policy receives two weighted cues at tick zero and a secondary cue after reaching waypoint one.',
        }[family]
        self._frame = dict(schema='aether.live.frame.v1', sequence=self.sequence,
            episode=self.episode, seed=self.seed,
            tick=max(self.candidate_world.timestep, self.control_world.timestep),
            done=self.candidate_world.done and self.control_world.done,
            settings=deepcopy(self._settings), provenance=deepcopy(self._provenance),
            provenance_epoch=self.provenance_epoch, model=deepcopy(self._model_info),
            topology={'links': deepcopy(self._links)},
            task=dict(family=family, **self._task_info(self.candidate_world)),
            plasticity=dict(**deepcopy(self._structural_report or {}),
                observed_ticks_available=len(self._activity_history),
                can_rewire=self.model.protocol != 'scalar' and len(self._activity_history) >= 2 and bool(self._links),
                comparison_scope='Structural edits affect the shared model in both branches; the clean control isolates channel interventions.'),
            candidate=deepcopy(candidate), control=deepcopy(control),
            comparison=dict(effective_action_differences=self._effective_differences,
                waiting_action_differences=self._waiting_differences,
                position_distance=float(np.linalg.norm(self.candidate_world.position - self.control_world.position)),
                return_delta=candidate['return'] - control['return']),
            history=list(self._history),
            payload_accounting='FP16 payload only: attempted before bandwidth, admitted before loss, surviving after loss, delivered after latency; excludes headers/scalar route. Link/mode energy is mean square of receiver-input packets.',
            observation_boundary='Goal coordinates are spectator metadata. ' + boundary)

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
        if self.model.protocol == 'scalar':
            raise ValueError('scalar protocol has no auxiliary regional links')
        existing = {tuple(link) for link in self._links}
        if (not isinstance(severed_pairs, (list, tuple)) or not severed_pairs
                or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                       or any(type(value) is not int for value in pair)
                       or tuple(pair) not in existing for pair in severed_pairs)
                or len({tuple(pair) for pair in severed_pairs}) != len(severed_pairs)):
            raise ValueError('lesion requires unique existing directed region links')
        replacement = self._copy_model(self.model)
        new_top = lesion_region_links(replacement.topology, severed_pairs)
        reconcile_model_topology(replacement, new_top)
        report = dict(operation='lesion', region_links_pruned=len(severed_pairs),
                      region_links_added=0, severed_links=list(map(list, severed_pairs)))
        return self._replace_structural(replacement, report)

    def rewire_hebbian(self):
        from aether.neurofabric.plasticity import HebbianPlasticityEngine
        if self.model.protocol == 'scalar':
            raise ValueError('scalar protocol has no auxiliary regional links')
        if len(self._activity_history) < 2:
            raise ValueError('rewiring requires at least two executed activity ticks in the current episode')
        engine = HebbianPlasticityEngine(pruning_percentile=50.0, rewiring_budget=2)
        activity = np.asarray(self._activity_history, dtype=np.float64)
        replacement = self._copy_model(self.model)
        report = asdict(engine.adapt_region_links(replacement, activity))
        report.update(operation='rewire', activity_source='executed_regional_activity',
                      observed_ticks=len(activity), episode_seed=self.seed,
                      task_family=self._settings['task_family'])
        return self._replace_structural(replacement, report)

    def _replace_structural(self, replacement, report):
        provenance = dict(self.provenance, kind='structurally_adapted',
                          label='Structural adaptation · exploratory',
                          source_digest=self.model.state_digest(), operation=report['operation'])
        return self.replace_model(replacement, provenance, structural_report=report)

    def replace_model(self, model, provenance=None, *, structural_report=None):
        replacement = self._copy_model(model)
        copied_provenance = deepcopy(provenance or {'kind': 'session-adapted'})
        self.model, self._provenance = replacement, copied_provenance
        self._structural_report = deepcopy(structural_report)
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
        if action is not None:
            self._activity_history.append([row['activity'] for row in self._branches['candidate']['regions']])
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
