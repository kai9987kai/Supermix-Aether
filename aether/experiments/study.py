"""A study runner that keeps training, executed query outcomes and claims distinct."""

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Integral
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import torch
from torch.nn import functional as F

from aether.config import ModelConfig
from aether.environments.stress import TASK_FAMILIES, create_world, stress_expert_batch
from aether.evidence.promotion import promotion_decision
from aether.evidence.receipts import write_receipt, write_source_snapshot
from aether.evidence.statistics import paired_summary
from aether.kernel.runtime import TissueModel
from aether.kernel.conformance import check_transport_conformance


PROTOCOLS = ('scalar', 'scalar_matched', 'vector', 'attention', 'phase', 'modes', 'wave')
INTERVENTIONS = ('zero_gain', 'erase_packets', 'shuffle_phase', 'random_phase')


@dataclass
class StudyConfig:
    seeds: list[int] = field(default_factory=lambda: [11, 29, 47])
    protocols: list[str] = field(default_factory=lambda: list(PROTOCOLS))
    updates: int = 40
    batch_size: int = 8
    evaluation_episodes: int = 24
    learning_rate: float = 0.003
    delay: int = 3
    horizon: int = 9
    retention_delay: int = 5
    retention_margin: float = 0.05
    min_confirmatory_seeds: int = 30
    model: dict = field(default_factory=dict)
    task_family: str = 'straight'

    def __post_init__(self):
        for key in ('updates', 'batch_size', 'evaluation_episodes', 'delay', 'horizon',
                    'retention_delay', 'min_confirmatory_seeds'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.min_confirmatory_seeds < 10:
            raise ValueError('min_confirmatory_seeds must be at least 10')
        if not isinstance(self.seeds, list) or not self.seeds:
            raise ValueError('seeds must be a nonempty list')
        if any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s <= 9999 for s in self.seeds):
            raise ValueError('seeds must be unique integers in [0,9999]')
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError('independent training seeds must be unique')
        if (not isinstance(self.protocols, list) or not self.protocols
                or any(p not in PROTOCOLS for p in self.protocols)
                or len(set(self.protocols)) != len(self.protocols)
                or 'scalar' not in self.protocols or 'wave' not in self.protocols):
            raise ValueError('protocols must be unique known names including scalar and wave')
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float))
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError('learning_rate must be finite and positive')
        if (isinstance(self.retention_margin, bool) or not isinstance(self.retention_margin, (int, float))
                or not math.isfinite(self.retention_margin) or not 0 <= self.retention_margin <= 1):
            raise ValueError('retention_margin must be finite in [0,1]')
        if self.retention_delay <= self.delay or self.horizon < self.delay + 2:
            raise ValueError('retention delay must exceed support delay; horizon must permit goal completion')
        if self.task_family not in TASK_FAMILIES:
            raise ValueError(f'task_family must be one of {TASK_FAMILIES}')
        # Validate the selected world's horizon before creating a run directory.
        create_world(self.task_family, seed=0, delay=self.delay, horizon=self.horizon)
        if self.updates * self.batch_size >= 100_000:
            raise ValueError('support episode budget exceeds disjoint seed block')
        if self.evaluation_episodes >= 1000:
            raise ValueError('evaluation_episodes must be below 1000 for disjoint seed blocks')
        if not isinstance(self.model, dict):
            raise ValueError('model must be an object')
        model = ModelConfig(**self.model)
        if model.input_dimension != 12 or model.action_count != 7:
            raise ValueError('navigation requires 12 observations and 7 actions')


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def episode_seeds(seed: int, count: int, retention: bool = False) -> list[int]:
    return [(2_000_000_000 if retention else 1_000_000_000) + seed * 1000 + i for i in range(count)]


def evaluate(model: TissueModel, seeds: list[int], *, delay: int, horizon: int,
             intervention: str = 'none', disturbance_seed: int = 0,
             task_family: str = 'straight') -> dict:
    digest = model.state_digest()
    was_training = model.training
    model.eval()
    records = []
    with torch.no_grad():
        for episode_seed in seeds:
            world = create_world(task_family, seed=episode_seed, delay=delay, horizon=horizon)
            state = model.initial_state(1)
            actions, total_reward = [], 0.0
            while not world.done:
                observation = torch.from_numpy(world.observe()[None])
                logits, state = model.step(observation, state, intervention=intervention,
                                           disturbance_seed=disturbance_seed + episode_seed + world.timestep)
                if not torch.isfinite(logits).all() or not torch.isfinite(state).all():
                    raise RuntimeError('query produced nonfinite neural state')
                action = int(logits.argmax(-1).item())
                _, reward, _ = world.step(action)
                actions.append(action)
                total_reward += reward
            records.append({'seed': episode_seed, 'success': world.success,
                            'return': round(total_reward, 8), 'actions': actions,
                            'final_position': world.position.tolist()})
    model.train(was_training)
    if model.state_digest() != digest:
        raise RuntimeError('query mutated learned parameters or persistent buffers')
    return {'success_rate': float(np.mean([r['success'] for r in records])),
            'mean_return': float(np.mean([r['return'] for r in records])),
            'executed_steps': sum(len(r['actions']) for r in records),
            'episode_seeds': list(seeds), 'episodes': records}


def train_one(config: StudyConfig, protocol: str, seed: int, output: Path) -> dict:
    model = TissueModel(ModelConfig(**config.model), protocol=protocol, seed=seed)
    conformance = check_transport_conformance(model)
    if not conformance['valid']:
        raise RuntimeError('tensor transport did not match packet reference')
    initial_digest = model.state_digest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    curve, executed = [], 0
    started = time.perf_counter()
    for update in range(config.updates):
        # Each seed owns a nonoverlapping support block well below query blocks.
        support_seeds = [seed * 100_000 + update * config.batch_size + i
                         for i in range(config.batch_size)]
        batch = stress_expert_batch(support_seeds, task_family=config.task_family,
                                    delay=config.delay, horizon=config.horizon)
        observation = torch.from_numpy(batch['observations'])
        targets = torch.from_numpy(batch['actions'])
        mask = torch.from_numpy(batch['mask'])
        optimizer.zero_grad(set_to_none=True)
        logits = model(observation, disturbance_seed=seed * 1000 + update)
        loss = F.cross_entropy(logits[mask], targets[mask])
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite support loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        executed += batch['executed_steps']
        curve.append({'update': update + 1, 'loss': float(loss.detach()),
                      'gradient_norm': float(norm),
                      'support_action_accuracy': float((logits[mask].argmax(-1) == targets[mask]).float().mean())})
    train_seconds = time.perf_counter() - started
    trained_digest = model.state_digest()
    evaluation = evaluate(model, episode_seeds(seed, config.evaluation_episodes),
                          delay=config.delay, horizon=config.horizon, task_family=config.task_family)
    retention_horizon = config.horizon + config.retention_delay - config.delay
    retention = evaluate(model, episode_seeds(seed, config.evaluation_episodes, retention=True),
                         delay=config.retention_delay, horizon=retention_horizon,
                         task_family=config.task_family)
    interventions = {}
    if protocol == 'wave':
        for name in INTERVENTIONS:
            interventions[name] = evaluate(model, episode_seeds(seed, config.evaluation_episodes),
                                            delay=config.delay, horizon=config.horizon,
                                            intervention=name, disturbance_seed=seed + 91_000,
                                            task_family=config.task_family)
    filename = f'checkpoints/{protocol}-{seed}.pt'
    (output / 'checkpoints').mkdir(exist_ok=True)
    checkpoint = {'schema': 'aether.checkpoint.v1', 'model_config': asdict(model.config),
                  'protocol': protocol, 'seed': seed, 'state_dict': model.state_dict(),
                  'trained_digest': trained_digest, 'evaluation': evaluation,
                  'retention': retention, 'interventions': interventions,
                  'delay': config.delay, 'horizon': config.horizon,
                  'retention_delay': config.retention_delay,
                  'retention_horizon': retention_horizon,
                  'task_family': config.task_family,
                  'intervention_disturbance_seed': seed + 91_000}
    torch.save(checkpoint, output / filename)
    replay = replay_checkpoint(output / filename)
    if not replay['valid']:
        raise RuntimeError(f'checkpoint failed exact local replay: {filename}')
    count = sum(p.numel() for p in model.parameters())
    return {'protocol': protocol, 'seed': seed, 'initial_digest': initial_digest,
            'trained_digest': trained_digest, 'query_immutable': model.state_digest() == trained_digest,
            'checkpoint': filename, 'replay_valid': replay['valid'], 'parameter_count': count,
            'train_seconds': train_seconds, 'executed_support_steps': executed,
            'resources': model.resource_metrics(), 'transport_conformance': conformance,
            'training_updates': config.updates, 'training_curve': curve,
            'evaluation': evaluation, 'retention': retention, 'interventions': interventions}


def replay_checkpoint(path: Path) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema') != 'aether.checkpoint.v1':
        raise ValueError('unsupported checkpoint schema')
    model = TissueModel(ModelConfig(**saved['model_config']), saved['protocol'], saved['seed'])
    model.load_state_dict(saved['state_dict'], strict=True)
    task_family = saved.get('task_family', 'straight')
    evaluation = evaluate(model, saved['evaluation']['episode_seeds'],
                          delay=saved['delay'], horizon=saved['horizon'], task_family=task_family)
    retention = evaluate(model, saved['retention']['episode_seeds'],
                         delay=saved['retention_delay'], horizon=saved['retention_horizon'],
                         task_family=task_family)
    interventions = {name: evaluate(model, outcome['episode_seeds'],
                                    delay=saved['delay'], horizon=saved['horizon'],
                                    intervention=name,
                                    disturbance_seed=saved['intervention_disturbance_seed'],
                                    task_family=task_family)
                     for name, outcome in saved['interventions'].items()}
    expected_names = set(INTERVENTIONS) if saved['protocol'] == 'wave' else set()
    valid = (evaluation == saved['evaluation'] and retention == saved['retention']
             and interventions == saved['interventions'] and set(interventions) == expected_names
             and model.state_digest() == saved['trained_digest'])
    return {'valid': valid, 'evaluation': evaluation, 'retention': retention,
            'interventions': interventions, 'state_digest': model.state_digest()}


def run_study(config: StudyConfig, output: Path, progress=None) -> dict:
    config.__post_init__()  # reject mutation of a previously validated config
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    write_json(output / 'config.json', asdict(config))
    source_root = Path(__file__).resolve().parents[2]
    # Snapshot and fingerprint share one allowlist and the exact archived bytes.
    fingerprint = write_source_snapshot(source_root, output / 'source-snapshot.zip')
    metadata = {'source_fingerprint': fingerprint,
                'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                'torch': str(torch.__version__), 'numpy': np.__version__,
                                'scipy': scipy.__version__, 'torch_threads': 1},
                'reproducibility_scope': 'exact local replay on recorded software and hardware'}
    runs = []
    for seed in config.seeds:
        for protocol in config.protocols:
            if progress:
                progress(f'training {protocol}, seed {seed}')
            run = train_one(config, protocol, seed, output)
            runs.append(run)
            write_json(output / 'partial-results.json', {'metadata': metadata, 'runs': runs})
            if progress:
                progress(f"{protocol} seed {seed}: success {run['evaluation']['success_rate']:.3f}, "
                         f"loss {run['training_curve'][0]['loss']:.3f} -> {run['training_curve'][-1]['loss']:.3f}")
    by_protocol = {name: sorted((r for r in runs if r['protocol'] == name), key=lambda r: r['seed'])
                   for name in config.protocols}
    scores = lambda name, split='evaluation': [r[split]['success_rate'] for r in by_protocol[name]]
    comparisons = {name: (scores('wave'), scores(name)) for name in config.protocols if name != 'wave'}
    promotion = promotion_decision(comparisons, retention_candidate=scores('wave', 'retention'),
                                   retention_baseline=scores('scalar', 'retention'),
                                   retention_margin=config.retention_margin,
                                   replay_valid=all(r['replay_valid'] for r in runs),
                                   integrity_valid=all(r['query_immutable'] for r in runs),
                                   resources_comparable=False,
                                   min_confirmatory_seeds=config.min_confirmatory_seeds)
    # Missing predeclared controls cannot be waived by narrowing the CLI arm list.
    if set(config.protocols) != set(PROTOCOLS):
        promotion['promoted'] = False
        promotion['status'] = 'rejected'
        promotion['reasons'].append('full predeclared seven-protocol comparison was not executed')
    aggregates = {name: {'mean_success': float(np.mean(scores(name))),
                          'mean_retention_success': float(np.mean(scores(name, 'retention'))),
                          'parameter_count': by_protocol[name][0]['parameter_count'],
                          'mean_train_seconds': float(np.mean([r['train_seconds'] for r in by_protocol[name]]))}
                  for name in config.protocols}
    task_limitations = {
        'straight': 'Disjoint query seeds draw from the same six fixed goal tasks; there are no held-out geometries.',
        'distraction': ('Disjoint query seeds resample six fixed goals with seeded sensory distractors; '
                        'there are no held-out geometries or demonstrated phase-filtering advantage.'),
        'waypoint': ('Disjoint query seeds resample 30 fixed ordered direction pairs; the second direction '
                     'is cued again after the first waypoint. There are no held-out route families.'),
    }
    result = {'schema': 'aether.study.v1', 'status': 'exploratory', 'metadata': metadata,
              'config': asdict(config), 'runs': runs, 'aggregates': aggregates,
              'paired_effects': {name: paired_summary(*pair) for name, pair in comparisons.items()},
              'promotion': promotion,
              'limitations': ['Synthetic delayed-cue imitation task, not reinforcement learning or general intelligence.',
                              'Training examples and update counts are matched; measured compute is not established equivalent.',
                              'Packet-reference impairments are separately tested; this training study uses clean tensor transport.',
                              task_limitations[config.task_family],
                              'Longer-delay evaluation is a retention proxy, not a sequential catastrophic-forgetting study.']}
    write_json(output / 'results.json', result)
    from aether.observatory import render_report
    render_report(result, output / 'index.html')
    write_receipt(output, metadata=metadata)
    return result
