"""Bounded, isolated imitation adaptation for the exploratory live workbench."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from numbers import Integral
from pathlib import Path
import threading
import time
from uuid import uuid4

import torch
from torch.nn import functional as F

from aether.environments.stress import TASK_FAMILIES, stress_expert_batch
from aether.evidence.receipts import source_fingerprint, write_receipt
from aether.experiments.study import evaluate
from aether.live.models import clone_model


def retention_eligible(before: float, after: float, margin: float = .05) -> bool:
    for value in (before, after, margin):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('retention scores and margin must be finite numbers in [0,1]')
    return after >= before - margin


class _Cancelled(Exception):
    pass


class TrainingJob:
    """One asynchronous candidate at a time; never updates its source model.

    Ready candidates are eligible only under the local retention margin. This
    eligibility is not a statistical promotion or confirmation of improvement.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._thread = None
        self._stop = threading.Event()
        self._candidate_model = None
        self._state = {'status': 'idle', 'progress': 0, 'total': 0, 'curve': [],
                       'comparison': None, 'can_apply': False, 'candidate': None,
                       'source_digest': None, 'task_delay': None, 'task_family': None, 'error': None,
                       'output_directory': None, 'label': 'Exploratory imitation sandbox'}

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy(self._state)

    @property
    def candidate_model(self):
        with self._lock:
            return self._candidate_model if self._state['status'] == 'ready' else None

    @property
    def can_apply(self) -> bool:
        with self._lock:
            return self._state['status'] == 'ready' and self._state['can_apply']

    def start(self, model, updates: int, output_root, delay: int = 3, task_family: str = 'straight'):
        if isinstance(updates, bool) or not isinstance(updates, Integral) or not 1 <= updates <= 100:
            raise ValueError('updates must be an integer in [1,100]')
        if isinstance(delay, bool) or not isinstance(delay, Integral) or not 1 <= delay <= 12:
            raise ValueError('delay must be an integer in [1,12]')
        if not isinstance(task_family, str) or task_family not in TASK_FAMILIES:
            raise ValueError(f'task_family must be one of {TASK_FAMILIES}')
        if model.config.input_dimension != 12 or model.config.action_count != 7:
            raise ValueError('live training requires the navigation observation/action interface')
        root = Path(output_root).resolve()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError('a candidate job is already running')
            source_digest = model.state_digest()
            baseline = clone_model(model).eval()
            candidate = clone_model(model)
            identifier = 'candidate-' + uuid4().hex[:16]
            directory = root / identifier
            directory.mkdir(parents=True, exist_ok=False)
            self._stop = threading.Event()
            self._candidate_model = None
            self._state = {'status': 'running', 'progress': 0, 'total': int(updates),
                           'curve': [], 'comparison': None, 'can_apply': False,
                           'candidate': None, 'source_digest': source_digest,
                           'task_delay': int(delay), 'task_family': task_family, 'error': None,
                           'output_directory': str(directory),
                           'label': 'Exploratory imitation sandbox'}
            torch.set_num_threads(1)
            self._thread = threading.Thread(target=self._run,
                                            args=(baseline, candidate, int(updates), int(delay),
                                                  task_family, directory, identifier),
                                            name='aether-candidate-training', daemon=True)
            self._thread.start()
        return self.snapshot()

    def _update(self, **values):
        with self._lock:
            self._state.update(deepcopy(values))

    def _check_cancelled(self):
        if self._stop.is_set():
            raise _Cancelled()

    def _evaluate(self, model, delay, task_family, query_seeds, retention_seeds):
        matrix = {}
        for family in TASK_FAMILIES:
            self._check_cancelled()
            query = evaluate(model, query_seeds, delay=delay, horizon=delay + 6,
                             task_family=family)
            self._check_cancelled()
            retention = evaluate(model, retention_seeds, delay=delay + 2, horizon=delay + 8,
                                 task_family=family)
            matrix[family] = {'query': query, 'retention': retention}
        self._check_cancelled()
        return dict(matrix[task_family], task_matrix=matrix)

    def _run(self, baseline, candidate, updates, delay, task_family, directory, identifier):
        started = time.perf_counter()
        source_root = Path(__file__).resolve().parents[2]
        record = {'schema': 'aether.live.training.v1', 'identifier': identifier,
                  'status': 'running', 'protocol': candidate.protocol,
                  'model_config': asdict(candidate.config), 'model_seed': candidate.seed,
                  'source_digest': baseline.state_digest(), 'task_delay': delay,
                  'task_family': task_family,
                  'updates_requested': updates, 'learning_rate': .003, 'batch_size': 8,
                  'support_seeds': [], 'executed_support_steps': 0, 'curve': [],
                  'promoted': False,
                  'limitations': ['Exploratory interactive imitation, not reinforcement learning.',
                                  'Repeated query feedback is not a fresh confirmatory study.',
                                  'Twelve shared seeds per condition sample fixed task families; not new geometries.',
                                  'Cross-task retention is a local diagnostic on a fixed task set, not statistical advancement.',
                                  'Waypoint stage two includes a visible secondary direction cue.',
                                  'Local retention eligibility does not establish statistical superiority.']}
        try:
            self._check_cancelled()
            metadata = {'source_fingerprint': source_fingerprint(source_root),
                        'purpose': 'exploratory live adaptation; no formal promotion',
                        'task_family': task_family, 'task_delay': delay}
            offset = candidate.seed * 1000
            query_seeds = [4_000_000_000 + offset + i for i in range(12)]
            retention_seeds = [5_000_000_000 + offset + i for i in range(12)]
            before = self._evaluate(baseline, delay, task_family, query_seeds, retention_seeds)
            record['before'] = before
            comparison = {'before': {'query_success': before['query']['success_rate'],
                                     'retention_success': before['retention']['success_rate']},
                          'after': None,
                          'task_matrix': {
                              family: {'before': {
                                  'query_success': scores['query']['success_rate'],
                                  'retention_success': scores['retention']['success_rate']},
                                  'after': None, 'retention_passed': None}
                              for family, scores in before['task_matrix'].items()}}
            self._update(comparison=comparison)
            optimizer = torch.optim.AdamW(candidate.parameters(), lr=.003)
            candidate.train()
            for update in range(updates):
                self._check_cancelled()
                seeds = [3_000_000_000 + candidate.seed * 10000 + update * 8 + i for i in range(8)]
                batch = stress_expert_batch(seeds, task_family=task_family, delay=delay, horizon=delay + 6)
                observations = torch.from_numpy(batch['observations'])
                targets, mask = torch.from_numpy(batch['actions']), torch.from_numpy(batch['mask'])
                optimizer.zero_grad(set_to_none=True)
                logits = candidate(observations, disturbance_seed=update + candidate.seed * 1000)
                loss = F.cross_entropy(logits[mask], targets[mask])
                if not torch.isfinite(loss):
                    raise RuntimeError('candidate produced nonfinite support loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                point = {'update': update + 1, 'loss': float(loss.detach()),
                         'accuracy': float((logits[mask].argmax(-1) == targets[mask]).float().mean())}
                record['curve'].append(point)
                record['support_seeds'].extend(seeds)
                record['executed_support_steps'] += batch['executed_steps']
                self._update(progress=update + 1, curve=record['curve'])
            candidate.eval()
            self._update(status='evaluating')
            after = self._evaluate(candidate, delay, task_family, query_seeds, retention_seeds)
            record['after'] = after
            if baseline.state_digest() != record['source_digest']:
                raise RuntimeError('baseline changed during isolated training')
            self._check_cancelled()
            retention_gates = {}
            for family in TASK_FAMILIES:
                before_scores, after_scores = before['task_matrix'][family], after['task_matrix'][family]
                retention_gates[family] = retention_eligible(
                    before_scores['retention']['success_rate'], after_scores['retention']['success_rate'])
                comparison['task_matrix'][family].update(
                    after={'query_success': after_scores['query']['success_rate'],
                           'retention_success': after_scores['retention']['success_rate']},
                    retention_passed=retention_gates[family])
            eligible = all(retention_gates.values())
            comparison['after'] = {'query_success': after['query']['success_rate'],
                                   'retention_success': after['retention']['success_rate']}
            record.update(status='ready', trained_digest=candidate.state_digest(),
                          can_apply=eligible, comparison=comparison, retention_gates=retention_gates,
                          elapsed_seconds=time.perf_counter() - started)
            checkpoint = {'schema': 'aether.live.candidate.v1', 'model_config': asdict(candidate.config),
                          'protocol': candidate.protocol, 'seed': candidate.seed,
                          'state_dict': candidate.state_dict(), 'trained_digest': candidate.state_digest(),
                          'source_digest': record['source_digest'], 'task_delay': delay,
                          'task_family': task_family}
            torch.save(checkpoint, directory / 'candidate.pt')
            self._save_record(directory, record, metadata)
            with self._lock:
                if self._stop.is_set():
                    # The completed record still describes the candidate, but
                    # closing a job never makes it applicable in this process.
                    self._state.update(status='cancelled', candidate=None, can_apply=False)
                else:
                    self._candidate_model = candidate
                    self._state.update(status='ready', comparison=comparison, can_apply=eligible,
                                       candidate={'id': identifier, 'can_apply': eligible,
                                                  'task_family': task_family,
                                                  'label': f'{candidate.protocol} · {task_family} · {updates} live updates',
                                                  'path': str(directory / 'candidate.pt')})
        except _Cancelled:
            record.update(status='cancelled', can_apply=False, elapsed_seconds=time.perf_counter() - started)
            self._save_stopped(directory, record, source_root)
            self._update(status='cancelled', can_apply=False, candidate=None)
        except Exception as error:
            record.update(status='error', error=str(error), can_apply=False,
                          elapsed_seconds=time.perf_counter() - started)
            self._save_stopped(directory, record, source_root)
            self._update(status='error', error=str(error), can_apply=False, candidate=None)

    @staticmethod
    def _save_record(directory, record, metadata):
        with (directory / 'training.json').open('x', encoding='utf-8') as target:
            json.dump(record, target, indent=2, allow_nan=False)
            target.write('\n')
        write_receipt(directory, metadata=metadata)

    def _save_stopped(self, directory, record, source_root):
        try:
            if not (directory / 'training.json').exists():
                self._save_record(directory, record, {'source_fingerprint': source_fingerprint(source_root)})
        except (OSError, ValueError):
            # The in-memory error remains available if evidence storage failed.
            pass

    def cancel(self, wait: bool = True):
        with self._lock:
            worker = self._thread
            if worker is not None and worker.is_alive():
                self._stop.set()
        if wait and worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5)
        return self.snapshot()
