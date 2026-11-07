import numpy as np
import pytest

from aether.environments.navigation import NavigationWorld, expert_batch


def test_cue_disappears_and_actions_change_position():
    world = NavigationWorld(seed=13, delay=2, horizon=7)
    first = world.observe()
    assert first.shape == (12,)
    assert first[:6].sum() == 1
    world.step(0)
    assert np.array_equal(world.position, np.zeros(3))  # waiting gate
    assert not world.observe()[:6].any()
    world.step(6)
    world.step(0)
    assert np.array_equal(world.position, [1, 0, 0])


def test_expert_succeeds_and_world_clones_are_independent():
    for seed in range(20):
        world = NavigationWorld(seed=seed)
        clone = world.clone()
        while not world.done:
            world.step(world.expert_action())
        assert world.success
        assert clone.timestep == 0
        assert np.array_equal(clone.position, np.zeros(3))


def test_expert_batch_is_executed_reproducible_and_masked():
    a = expert_batch([1, 2, 3], delay=3, horizon=8)
    b = expert_batch([1, 2, 3], delay=3, horizon=8)
    for key in ('observations', 'actions', 'mask'):
        assert np.array_equal(a[key], b[key])
    assert a['observations'].shape == (3, 8, 12)
    assert not a['mask'][:, :3].any()
    assert a['mask'].any()
    assert a['executed_steps'] > 0


@pytest.mark.parametrize('kwargs', [{'delay': -1}, {'horizon': 3}, {'distance': 0}, {'radius': 1}])
def test_invalid_task_parameters(kwargs):
    with pytest.raises(ValueError):
        NavigationWorld(seed=1, **kwargs)


def test_invalid_action_does_not_change_world():
    world = NavigationWorld(seed=1)
    for action in (7, -1, 0.5, True):
        with pytest.raises(ValueError):
            world.step(action)
    assert world.timestep == 0
