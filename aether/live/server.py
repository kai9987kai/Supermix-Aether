"""A bounded, loopback-only HTTP/SSE coordinator for one shared live lab.

GET requests consume cached frames. Only the application's single scheduler or
an explicit control can execute the neural session.
"""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import re
import threading
import time
from urllib.parse import unquote, urlsplit

import torch

from aether.config import ModelConfig
from aether.kernel.runtime import PROTOCOLS, TissueModel
from aether.live.session import LiveSession
from aether.live.training import TrainingJob


_CHECKPOINT = re.compile(r'(' + '|'.join(PROTOCOLS) + r')-([0-9]{1,4})\.pt\Z')
_MAX_REQUEST = 16 * 1024
_BUSY = {'running', 'evaluating', 'training'}


def _safe_file(root, relative):
    """Resolve a regular file without following any link below the root."""
    if not isinstance(relative, str) or '\\' in relative or '\x00' in relative:
        raise ValueError('invalid file path')
    parts = relative.split('/')
    if not parts or any(part in ('', '.', '..') or ':' in part for part in parts):
        raise ValueError('invalid file path')
    target = root
    for part in parts:
        target = target / part
        if target.is_symlink() or getattr(target, 'is_junction', lambda: False)():
            raise ValueError('linked files are not served')
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError('file is outside configured directory')
    return resolved


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum},{maximum}]')
    return value


class LiveApplication:
    def __init__(self, study_dir, output_root=None):
        torch.set_num_threads(1)
        self.study_dir = Path(study_dir).resolve(strict=True)
        if not self.study_dir.is_dir():
            raise ValueError('study directory must be a directory')
        self.output_root = (Path(output_root) if output_root is not None
                            else self.study_dir.parent / 'live-training').resolve()
        if self.output_root.is_relative_to(self.study_dir):
            raise ValueError('training output must be outside the recorded study')
        self._catalog = {}
        checkpoint_root = self.study_dir / 'checkpoints'
        if checkpoint_root.is_dir() and not checkpoint_root.is_symlink():
            for candidate in sorted(checkpoint_root.rglob('*.pt')):
                match = _CHECKPOINT.fullmatch(candidate.name)
                if not match:
                    continue
                relative = candidate.relative_to(self.study_dir).as_posix()
                try:
                    safe = _safe_file(self.study_dir, relative)
                except (OSError, ValueError):
                    continue
                checkpoint_id = candidate.relative_to(checkpoint_root).with_suffix('').as_posix()
                self._catalog[checkpoint_id] = {
                    'id': checkpoint_id, 'label': f'{match[1]} · recorded seed {int(match[2])}',
                    'protocol': match[1], 'seed': int(match[2]), 'path': safe,
                    'checkpoint': relative}
        if not self._catalog:
            raise ValueError('study contains no recognized checkpoints')
        self.selected_checkpoint = 'wave-11' if 'wave-11' in self._catalog else next(iter(self._catalog))
        model, provenance = self._load_checkpoint(self.selected_checkpoint)
        self.session = LiveSession(model, provenance=provenance)
        self.training = TrainingJob()
        self.playing, self.speed = False, 3
        self.stats = {'episodes': 0, 'successes': 0}
        self._counted_episode = None
        self._previous = None
        self._applied_candidate = None
        self._runtime_error = None
        self._revision = 0
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stopped = threading.Event()
        self._clients = 0
        self._frame = {}
        self._training_snapshot = self.training.snapshot()
        with self._lock:
            self._publish()
        self._scheduler = threading.Thread(target=self._run, name='aether-live-scheduler', daemon=True)
        self._scheduler.start()

    def _load_checkpoint(self, checkpoint_id):
        entry = self._catalog.get(checkpoint_id)
        if entry is None:
            raise ValueError('unknown checkpoint id')
        path = _safe_file(self.study_dir, entry['checkpoint'])
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('checkpoint exceeds live loading budget')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(saved, dict) or saved.get('schema') != 'aether.checkpoint.v1':
            raise ValueError('unsupported checkpoint schema')
        if saved.get('protocol') != entry['protocol'] or saved.get('seed') != entry['seed']:
            raise ValueError('checkpoint identity disagrees with its catalog filename')
        config = ModelConfig(**saved['model_config'])
        limits = ModelConfig()
        for name in ('neuron_count', 'scalar_edges', 'region_count', 'region_message_edges',
                     'state_dimension', 'mode_count', 'complex_dimensions'):
            if getattr(config, name) > getattr(limits, name):
                raise ValueError('checkpoint exceeds live model budget')
        if config.input_dimension != 12 or config.action_count != 7:
            raise ValueError('checkpoint does not match the navigation observation/action contract')
        model = TissueModel(config, entry['protocol'], entry['seed'])
        model.load_state_dict(saved['state_dict'], strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError('checkpoint contains nonfinite tensors')
        if saved.get('trained_digest') != model.state_digest():
            raise ValueError('checkpoint trained digest does not match its weights')
        model.eval()
        return model, {key: entry[key] for key in ('id', 'protocol', 'seed', 'checkpoint')} | {
            'kind': 'recorded', 'label': entry['label']}

    def _publish(self):
        """Called with the coordinator lock; snapshot is cached and copied for callers."""
        self._revision += 1
        frame = self.session.snapshot()
        frame['engine_sequence'] = frame['sequence']
        frame['sequence'] = self._revision
        training = deepcopy(self._training_snapshot)
        frame.update(playing=self.playing, speed=self.speed,
                     checkpoints=[{k: entry[k] for k in ('id', 'label', 'protocol', 'seed')}
                                  for entry in self._catalog.values()],
                     selected_checkpoint=self.selected_checkpoint,
                     training=training, stats=dict(self.stats),
                     can_rollback=self._previous is not None, runtime_error=self._runtime_error)
        self._frame = frame
        self._condition.notify_all()

    def _step(self):
        frame = self.session.step()
        identity = (frame['episode'], frame['seed'])
        if frame['done'] and identity != self._counted_episode:
            self.stats['episodes'] += 1
            self.stats['successes'] += int(frame['candidate']['success'])
            self._counted_episode = identity

    def _run(self):
        next_tick = time.monotonic()
        while not self._stopped.wait(.025):
            with self._lock:
                changed = False
                snapshot = self.training.snapshot()
                if snapshot != self._training_snapshot:
                    self._training_snapshot = snapshot
                    changed = True
                now = time.monotonic()
                if self.playing and snapshot.get('status') not in _BUSY and now >= next_tick:
                    try:
                        self._step()
                    except Exception as error:
                        self.playing = False
                        self._runtime_error = f'Live execution failed: {error}'
                    next_tick = time.monotonic() + 1 / self.speed
                    changed = True
                elif not self.playing:
                    next_tick = now
                if changed:
                    self._publish()

    def state(self):
        with self._lock:
            return deepcopy(self._frame)

    def control(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get('action'), str):
            raise ValueError('control requires an action object')
        fields = {'play': set(), 'pause': set(), 'step': set(), 'reset': {'seed'},
                  'configure': {'settings'}, 'checkpoint': {'id'}, 'speed': {'value'},
                  'train': {'updates'}, 'apply': set(), 'rollback': set(),
                  'lesion': {'region_pair'}, 'rewire': set()}
        action = payload['action']
        if action not in fields or set(payload) - {'action'} - fields[action]:
            raise ValueError('unknown action or action fields')
        with self._lock:
            if self._stopped.is_set():
                raise ValueError('live application is closed')
            training = self.training.snapshot()
            if training.get('status') in _BUSY and action != 'pause':
                raise ValueError('training is running; pause is the only available control')
            if action == 'play':
                self.playing = True
            elif action == 'pause':
                self.playing = False
            elif action == 'step':
                self._step()
                self.playing = False
            elif action == 'reset':
                seed = payload.get('seed')
                if seed is not None:
                    _integer(seed, 0, 2**53 - 1, 'seed')
                self.session.reset(seed=seed)
            elif action == 'configure':
                settings = payload.get('settings')
                if not isinstance(settings, dict):
                    raise ValueError('settings must be an object')
                self.session.configure(**settings)
            elif action == 'checkpoint':
                checkpoint_id = payload.get('id')
                if not isinstance(checkpoint_id, str):
                    raise ValueError('checkpoint id must be a string')
                model, provenance = self._load_checkpoint(checkpoint_id)
                self.session.replace_model(model, provenance=provenance)
                self.selected_checkpoint = checkpoint_id
                self._previous = None
            elif action == 'speed':
                value = payload.get('value')
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 8:
                    raise ValueError('speed must be finite in [1,8]')
                self.speed = value
            elif action == 'train':
                updates = _integer(payload.get('updates', 10), 1, 100, 'updates')
                output_root = self.output_root.resolve()
                if output_root.is_relative_to(self.study_dir):
                    raise ValueError('training output must be outside the recorded study')
                self.training.start(self.session.model, updates=updates, output_root=output_root,
                                    delay=self.session.snapshot()['settings']['delay'])
                self.playing = False
            elif action == 'apply':
                if training.get('status') != 'ready' or not self.training.can_apply or self.training.candidate_model is None:
                    raise ValueError('candidate is not ready or did not pass the retention gate')
                digest = self.session.model.state_digest()
                if training.get('source_digest') != digest or training.get('task_delay') != self.session.delay:
                    raise ValueError('candidate is stale for the active model or task delay')
                candidate_digest = self.training.candidate_model.state_digest()
                if candidate_digest == self._applied_candidate:
                    raise ValueError('candidate has already been applied')
                before = (self.session.model, deepcopy(self.session.snapshot()['provenance']), self.selected_checkpoint)
                provenance = {'kind': 'session_adapted', 'label': 'Session adapted · exploratory',
                              'protocol': self.training.candidate_model.protocol,
                              'source_checkpoint': self.selected_checkpoint,
                              'source_digest': digest, 'training': training.get('output_directory')}
                self.session.replace_model(self.training.candidate_model, provenance=provenance)
                self._previous, self._applied_candidate = before, candidate_digest
                self.playing = False
            elif action == 'rollback':
                if self._previous is None:
                    raise ValueError('no previous model is available')
                model, provenance, selected = self._previous
                self.session.replace_model(model, provenance=provenance)
                self.selected_checkpoint, self._previous = selected, None
                self.playing = False
            elif action == 'lesion':
                pair = payload.get('region_pair')
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError('region_pair must be a 2-element list [src, dst]')
                self.session.lesion_links([tuple(pair)])
                self.playing = False
            elif action == 'rewire':
                self.session.rewire_hebbian()
                self.playing = False
            self._training_snapshot = self.training.snapshot()
            self._runtime_error = None
            self._publish()
            return deepcopy(self._frame)

    def export(self):
        with self._lock:
            value = self.session.export()
            value['coordinator'] = {'playing': self.playing, 'speed': self.speed,
                                    'stats': dict(self.stats), 'revision': self._revision}
            return value

    def close(self):
        self._stopped.set()
        with self._condition:
            self.playing = False
            self._condition.notify_all()
        self.training.cancel()
        if threading.current_thread() is not self._scheduler:
            self._scheduler.join(timeout=5)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object key')
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError('nonfinite JSON values are not accepted')


class _LiveServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(application, host='127.0.0.1', port=8789):
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('live server must bind to IPv4 loopback')
    _integer(port, 0, 65535, 'port')
    static_root = Path(__file__).parent / 'web'

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def log_message(self, *args):
            pass

        def _send(self, status, value, content_type='application/json; charset=utf-8', extra=None):
            body = (json.dumps(value, allow_nan=False, separators=(',', ':')).encode('utf-8')
                    if content_type.startswith('application/json') else value)
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != 'HEAD':
                try:
                    self.wfile.write(body)
                except (OSError, TimeoutError):
                    self.close_connection = True

        def _host_allowed(self):
            hosts = self.headers.get_all('Host', [])
            if len(hosts) != 1:
                return False
            try:
                parsed = urlsplit('//' + hosts[0])
                return (parsed.hostname in ('127.0.0.1', 'localhost') and parsed.port == self.server.server_port
                        and not parsed.username and not parsed.password and not parsed.path and not parsed.query and not parsed.fragment)
            except ValueError:
                return False

        def _origin_allowed(self):
            origins = self.headers.get_all('Origin', [])
            if not origins:
                return True
            if len(origins) != 1:
                return False
            try:
                parsed = urlsplit(origins[0])
                return (parsed.scheme == 'http' and parsed.netloc == self.headers['Host']
                        and not parsed.path and not parsed.query and not parsed.fragment)
            except ValueError:
                return False

        def do_GET(self):
            if not self._host_allowed():
                self._send(403, {'error': 'loopback Host with the server port is required'})
                return
            path = urlsplit(self.path).path
            if path == '/api/state':
                self._send(200, application.state())
            elif path == '/api/events':
                self._events()
            elif path == '/api/export':
                self._send(200, application.export(), extra={'Content-Disposition': 'attachment; filename="aether-live-trace.json"'})
            elif path == '/favicon.ico':
                self._send(204, b'', 'image/x-icon')
            else:
                try:
                    if path in ('/', '/index.html', '/app.js', '/styles.css'):
                        file_path = _safe_file(static_root.resolve(), 'index.html' if path == '/' else path[1:])
                    elif path.startswith('/study/'):
                        file_path = _safe_file(application.study_dir, unquote(path[7:]) or 'index.html')
                    else:
                        raise ValueError('unknown path')
                    content_type = mimetypes.guess_type(file_path.name)[0] or 'application/octet-stream'
                    if content_type.startswith('text/') or content_type == 'application/javascript':
                        content_type += '; charset=utf-8'
                    self._send(200, file_path.read_bytes(), content_type)
                except (ValueError, OSError):
                    self._send(404, {'error': 'not found'})

        def do_POST(self):
            self.close_connection = True
            lengths = self.headers.get_all('Content-Length', [])
            length = int(lengths[0]) if (len(lengths) == 1 and lengths[0].isdigit()) else 0
            if not self._host_allowed() or not self._origin_allowed():
                if length:
                    try:
                        self.rfile.read(min(length, 65536))
                    except (OSError, TimeoutError):
                        pass
                self._send(403, {'error': 'same loopback origin is required'})
                return
            if self.path != '/api/control':
                if length:
                    try:
                        self.rfile.read(min(length, 65536))
                    except (OSError, TimeoutError):
                        pass
                self._send(404, {'error': 'not found'})
                return
            if self.headers.get_content_type() != 'application/json':
                if length:
                    try:
                        self.rfile.read(min(length, 65536))
                    except (OSError, TimeoutError):
                        pass
                self._send(415, {'error': 'Content-Type must be application/json'})
                return
            if self.headers.get('Transfer-Encoding') or len(lengths) != 1 or not lengths[0].isdigit():
                self._send(400, {'error': 'one Content-Length is required'})
                return
            if length > _MAX_REQUEST:
                try:
                    self.rfile.read(min(length, 65536))
                except (OSError, TimeoutError):
                    pass
                self._send(413, {'error': 'control body exceeds 16 KiB'})
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('incomplete request body')
                payload = json.loads(raw.decode('utf-8'), object_pairs_hook=_json_object,
                                     parse_constant=_bad_constant)
                self._send(200, application.control(payload))
            except (ValueError, TypeError, KeyError, OSError, RuntimeError, RecursionError) as error:
                self._send(400, {'error': str(error)})

        def _events(self):
            with application._condition:
                admitted = application._clients < 8
                if admitted:
                    application._clients += 1
            if not admitted:
                self._send(503, {'error': 'live stream client limit reached'})
                return
            try:
                self.connection.settimeout(2)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                revision = -1
                while not application._stopped.is_set():
                    with application._condition:
                        application._condition.wait_for(
                            lambda: application._revision != revision or application._stopped.is_set(), timeout=2)
                        if application._stopped.is_set():
                            break
                        current = application._revision
                        frame = application._frame if current != revision else None
                    if frame is None:
                        self.wfile.write(b': heartbeat\n\n')
                    else:
                        body = json.dumps(frame, allow_nan=False, separators=(',', ':'))
                        self.wfile.write(f'id: {current}\nevent: frame\ndata: {body}\n\n'.encode('utf-8'))
                        revision = current
                    self.wfile.flush()
            except (OSError, TimeoutError):
                pass
            finally:
                self.close_connection = True
                with application._condition:
                    application._clients -= 1

    server = _LiveServer((host, port), Handler)
    server.application = application
    return server


def serve_live(study_dir, *, output_root=None, host='127.0.0.1', port=8789):
    application = LiveApplication(study_dir, output_root=output_root)
    server = None
    try:
        server = create_server(application, host=host, port=port)
        print(f'AETHER Live Lab: http://{host}:{server.server_port}', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        application.close()
        if server is not None:
            server.server_close()
