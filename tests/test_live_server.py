"""Exercise the real loopback coordinator with inexpensive neural checkpoints."""
from dataclasses import asdict
import http.client
import json
import threading
import time

import pytest
import torch

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
from aether.live.server import LiveApplication, create_server


@pytest.fixture
def study(tmp_path):
    directory = tmp_path / 'study'
    (directory / 'checkpoints').mkdir(parents=True)
    config = ModelConfig(neuron_count=16, scalar_edges=32, region_count=4,
                         region_message_edges=8, state_dimension=4, mode_count=2,
                         complex_dimensions=2)
    model = TissueModel(config, 'wave', seed=11)
    torch.save({'schema': 'aether.checkpoint.v1', 'model_config': asdict(config),
                'protocol': 'wave', 'seed': 11, 'state_dict': model.state_dict(),
                'trained_digest': model.state_digest()}, directory / 'checkpoints' / 'wave-11.pt')
    (directory / 'index.html').write_text('<h1>Recorded evidence</h1>', encoding='utf-8')
    return directory


@pytest.fixture
def app(study, tmp_path):
    application = LiveApplication(study, output_root=tmp_path / 'training')
    yield application
    application.close()


@pytest.fixture
def http_server(app):
    server = create_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(2)


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
    if isinstance(body, dict):
        body = json.dumps(body)
    headers = {'Content-Type': 'application/json', **(headers or {})}
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
    except (http.client.RemoteDisconnected, ConnectionResetError):
        result = 413, b'{"error":"connection reset"}', {}
    finally:
        connection.close()
    return result


def test_paused_reads_do_not_advance_and_mutations_are_atomic(app):
    initial = app.state()
    assert initial['selected_checkpoint'] == 'wave-11'
    assert initial['playing'] is False
    assert initial['provenance']['kind'] == 'recorded'
    for _ in range(4):
        assert app.state()['tick'] == initial['tick']
    with pytest.raises(ValueError):
        app.control({'action': 'configure', 'settings': {'delay': 4, 'unknown': True}})
    assert app.state() == initial
    with pytest.raises(ValueError):
        app.control({'action': 'step', 'extra': 2})
    frame = app.control({'action': 'step'})
    assert frame['tick'] == initial['tick'] + 1
    assert frame['sequence'] > initial['sequence']
    assert app.export()['schema']


def test_one_scheduler_pause_and_episode_count(app):
    scheduler = app._scheduler
    app.control({'action': 'speed', 'value': 8})
    app.control({'action': 'play'})
    app.control({'action': 'play'})
    deadline = time.monotonic() + 3
    while app.state()['tick'] < 2 and time.monotonic() < deadline:
        time.sleep(.02)
    paused = app.control({'action': 'pause'})
    assert paused['tick'] >= 2
    time.sleep(.18)
    assert app.state()['tick'] == paused['tick']
    assert app._scheduler is scheduler
    app.control({'action': 'reset', 'seed': 999})
    for _ in range(20):
        frame = app.control({'action': 'step'})
        if frame['done']:
            break
    assert frame['done']
    assert frame['stats']['episodes'] == 1
    # The next executed tick restarts instead of counting a finished frame twice.
    next_frame = app.control({'action': 'step'})
    assert next_frame['seed'] == 1000
    assert next_frame['stats']['episodes'] == 1
    assert next_frame['sequence'] > frame['sequence']


def test_http_state_controls_events_and_readonly_report(http_server):
    status, data, headers = request(http_server, 'GET', '/api/state')
    assert status == 200 and headers['Cache-Control'] == 'no-store'
    initial = json.loads(data)
    status, data, _ = request(http_server, 'POST', '/api/control', {'action': 'step'})
    assert status == 200 and json.loads(data)['tick'] == initial['tick'] + 1
    connection = http.client.HTTPConnection('127.0.0.1', http_server.server_port, timeout=3)
    connection.request('GET', '/api/events')
    stream = connection.getresponse()
    assert stream.status == 200
    lines = [stream.readline().decode() for _ in range(4)]
    assert any(line == 'event: frame\n' for line in lines)
    frame = json.loads(next(line[6:] for line in lines if line.startswith('data: ')))
    assert frame['tick'] == initial['tick'] + 1
    stream.close()
    connection.close()
    assert request(http_server, 'GET', '/study/')[1] == b'<h1>Recorded evidence</h1>'
    status, body, _ = request(http_server, 'GET', '/api/export')
    assert status == 200 and json.loads(body)['schema']


@pytest.mark.parametrize('path', ['/study/../pyproject.toml', '/study/%2e%2e/pyproject.toml',
                                  '/study/%5c..%5cpyproject.toml', '/aether/cli.py'])
def test_static_allowlist(http_server, path):
    assert request(http_server, 'GET', path)[0] == 404


@pytest.mark.parametrize('body,headers', [
    ('{"action":"step","action":"play"}', {}),
    ('{"action":"speed","value":NaN}', {}),
    ('{"action":"step"}', {'Content-Type': 'text/plain'}),
    ('{"action":"step"}', {'Origin': 'https://example.com'}),
    ('{"action":"step"}', {'Host': 'example.com'}),
    ('{"action":"checkpoint","id":"../wave-11"}', {}),
    ('{"action":"play","settings":{}}', {}),
    ('x' * 16385, {}),
])
def test_invalid_http_requests_do_not_mutate(http_server, app, body, headers):
    before = app.state()
    assert request(http_server, 'POST', '/api/control', body, headers)[0] in (400, 403, 413, 415)
    assert app.state() == before


def test_checkpoint_manifest_identity_and_strict_load(study, tmp_path):
    saved = torch.load(study / 'checkpoints' / 'wave-11.pt', weights_only=True)
    saved['protocol'] = 'scalar'
    torch.save(saved, study / 'checkpoints' / 'wave-11.pt')
    with pytest.raises(ValueError, match='identity'):
        LiveApplication(study, tmp_path / 'training')


def test_training_busy_controls_and_apply_rollback(app):
    class Trainer:
        def __init__(self):
            self.status = 'idle'
            self.candidate_model = None
            self.can_apply = False

        def snapshot(self):
            return {'status': self.status, 'progress': 0, 'total': 1, 'curve': [],
                    'can_apply': self.can_apply, 'candidate': None, 'comparison': None, 'error': None,
                    'source_digest': getattr(self, 'source_digest', None), 'task_delay': 3}

        def start(self, model, updates, output_root, delay=3):
            import copy
            self.source_digest = model.state_digest()
            self.candidate_model = copy.deepcopy(model)
            with torch.no_grad():
                self.candidate_model.output_decoder.bias.add_(.01)
            self.status = 'training'

        def cancel(self):
            pass

    trainer = Trainer()
    app.training = trainer
    original = app.session.model.state_digest()
    app.control({'action': 'train', 'updates': 1})
    with pytest.raises(ValueError, match='training'):
        app.control({'action': 'configure', 'settings': {'delay': 5}})
    assert app.session.model.state_digest() == original
    trainer.status, trainer.can_apply = 'ready', True
    deadline = time.monotonic() + 2
    while app.state()['training']['status'] != 'ready' and time.monotonic() < deadline:
        time.sleep(.025)
    assert app.state()['training']['status'] == 'ready'
    assert app.state()['playing'] is False
    # A completed candidate belongs to the exact task delay it was trained on.
    app.control({'action': 'configure', 'settings': {'delay': 4}})
    with pytest.raises(ValueError, match='stale'):
        app.control({'action': 'apply'})
    app.control({'action': 'configure', 'settings': {'delay': 3}})
    applied = app.control({'action': 'apply'})
    assert applied['provenance']['kind'] == 'session_adapted'
    assert app.session.model.state_digest() != original
    restored = app.control({'action': 'rollback'})
    assert restored['provenance']['kind'] == 'recorded'
    assert app.session.model.state_digest() == original


def test_stream_client_limit_and_clean_close(http_server, app):
    connections = []
    responses = []
    try:
        for _ in range(8):
            connection = http.client.HTTPConnection('127.0.0.1', http_server.server_port, timeout=3)
            connection.request('GET', '/api/events')
            response = connection.getresponse()
            assert response.status == 200
            connections.append(connection)
            responses.append(response)
        assert request(http_server, 'GET', '/api/events')[0] == 503
        tick = app.state()['tick']
        assert app.control({'action': 'step'})['tick'] == tick + 1
        app.close()
        assert not app._scheduler.is_alive()
    finally:
        for response in responses:
            response.close()
        for connection in connections:
            connection.close()


def test_only_cataloged_files_can_load_and_output_stays_outside_study(study, tmp_path):
    (study / 'checkpoints' / 'unrecognized.pt').write_bytes(b'not a checkpoint')
    app = LiveApplication(study, tmp_path / 'training')
    try:
        assert [item['id'] for item in app.state()['checkpoints']] == ['wave-11']
    finally:
        app.close()
    with pytest.raises(ValueError, match='outside'):
        LiveApplication(study, output_root=study / 'training')
