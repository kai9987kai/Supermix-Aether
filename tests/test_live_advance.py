import json
import pytest
import torch

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
from aether.live.session import LiveSession
from aether.live.server import LiveApplication


def test_live_session_task_families():
    c = ModelConfig(neuron_count=16, scalar_edges=32, region_count=4,
                    region_message_edges=8, state_dimension=4, mode_count=2,
                    complex_dimensions=2)
    model = TissueModel(c, 'wave', seed=1)
    session = LiveSession(model)

    assert session.settings['task_family'] == 'straight'
    f0 = session.snapshot()
    assert f0['settings']['task_family'] == 'straight'
    assert f0['model']['step_flops'] > 0

    # Configure distraction task family
    session.configure(task_family='distraction')
    assert session.settings['task_family'] == 'distraction'
    f1 = session.step()
    assert f1['tick'] == 1

    # Configure waypoint task family
    session.configure(task_family='waypoint')
    assert session.settings['task_family'] == 'waypoint'
    f2 = session.step()
    assert f2['tick'] == 1

    # Invalid task family rejected
    with pytest.raises(ValueError, match='unknown task_family'):
        session.configure(task_family='impossible_task')


def test_live_session_structural_plasticity():
    c = ModelConfig(neuron_count=16, scalar_edges=32, region_count=4,
                    region_message_edges=8, state_dimension=4, mode_count=2,
                    complex_dimensions=2)
    model = TissueModel(c, 'wave', seed=1)
    session = LiveSession(model)

    initial_links = len(session.snapshot()['topology']['links'])
    first_pair = session.snapshot()['topology']['links'][0]

    # Lesion one directed link
    session.lesion_links([first_pair])
    post_lesion = session.snapshot()
    assert len(post_lesion['topology']['links']) == initial_links - 1

    # Step the lesioned model
    step_frame = session.step()
    assert step_frame['tick'] == 1

    # Trigger Hebbian structural rewiring
    session.step()
    session.rewire_hebbian()
    post_rewire = session.snapshot()
    assert 'hebbian' in post_rewire['model']['digest'] or post_rewire['sequence'] > 0


def test_live_application_lesion_and_rewire(tmp_path):
    study_dir = tmp_path / 'study'
    (study_dir / 'checkpoints').mkdir(parents=True)
    c = ModelConfig(neuron_count=16, scalar_edges=32, region_count=4,
                    region_message_edges=8, state_dimension=4, mode_count=2,
                    complex_dimensions=2)
    model = TissueModel(c, 'wave', seed=11)
    from dataclasses import asdict
    torch.save({'schema': 'aether.checkpoint.v1', 'model_config': asdict(c),
                'protocol': 'wave', 'seed': 11, 'state_dict': model.state_dict(),
                'trained_digest': model.state_digest()}, study_dir / 'checkpoints' / 'wave-11.pt')
    (study_dir / 'index.html').write_text('<h1>Recorded</h1>', encoding='utf-8')

    app = LiveApplication(study_dir, output_root=tmp_path / 'training')
    try:
        initial_frame = app.state()
        first_link = initial_frame['topology']['links'][0]

        # Lesion via control API
        lesioned = app.control({'action': 'lesion', 'region_pair': first_link})
        assert len(lesioned['topology']['links']) == len(initial_frame['topology']['links']) - 1

        # Rewire via control API
        app.control({'action': 'step'})
        app.control({'action': 'step'})
        rewired = app.control({'action': 'rewire'})
        assert rewired['tick'] == 0

        # Configure task family via control API
        configured = app.control({'action': 'configure', 'settings': {'task_family': 'distraction'}})
        assert configured['settings']['task_family'] == 'distraction'
    finally:
        app.close()
