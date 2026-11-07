import numpy as np
import pytest

from aether.environments.stress import (
    DistractionNavigationWorld,
    WaypointNavigationWorld,
    create_world,
    stress_expert_batch,
    TASK_FAMILIES,
)
from aether.environments.navigation import NavigationWorld


def test_task_families_enumeration():
    assert 'straight' in TASK_FAMILIES
    assert 'distraction' in TASK_FAMILIES
    assert 'waypoint' in TASK_FAMILIES


def test_create_world_factory():
    w_straight = create_world('straight', seed=42)
    assert isinstance(w_straight, NavigationWorld)
    assert not isinstance(w_straight, DistractionNavigationWorld)

    w_distract = create_world('distraction', seed=42)
    assert isinstance(w_distract, DistractionNavigationWorld)

    w_waypoint = create_world('waypoint', seed=42)
    assert isinstance(w_waypoint, WaypointNavigationWorld)

    with pytest.raises(ValueError, match='unknown task_family'):
        create_world('non_existent', seed=42)


def test_distraction_world_observation_contract_and_pulses():
    world = DistractionNavigationWorld(seed=10, delay=3, horizon=9)
    # At tick 0: true cue is presented
    obs0 = world.observe()
    assert obs0.shape == (12,)
    assert obs0[world.direction] == 1.0
    assert obs0[9] == 3.0 / 9.0  # delay remaining
    assert obs0[10] == 0.0  # movement not enabled

    # Step hold tick 0
    obs1, r1, done1 = world.step(6)
    assert not done1
    assert world.timestep == 1
    # Distractor pulse is active on some other direction channel
    distractor_dirs = [d for d in range(6) if d != world.direction]
    assert any(obs1[d] > 0.0 for d in distractor_dirs)
    # The true cue is NOT present at tick 1
    assert obs1[world.direction] == 0.0

    # Step hold tick 1
    obs2, _, _ = world.step(6)
    assert world.timestep == 2

    # Step hold tick 2 -> movement now enabled at tick 3
    obs3, _, _ = world.step(6)
    assert world.timestep == 3
    assert obs3[10] == 1.0  # movement enabled
    # Distractors should be 0 once movement starts
    assert all(obs3[d] == 0.0 for d in range(6))


def test_distraction_expert_reaches_goal():
    world = DistractionNavigationWorld(seed=123, delay=3, horizon=9)
    while not world.done:
        action = world.expert_action()
        world.step(action)
    assert world.success
    assert np.array_equal(world.position, world.goal)


def test_waypoint_world_two_stage_navigation():
    world = WaypointNavigationWorld(seed=77, delay=2, horizon=10, distance=2, radius=4)
    obs = world.observe()
    assert obs.shape == (12,)
    # Cue at tick 0 has both dir1 and dir2
    assert obs[world.dir1] == 1.0
    assert obs[world.dir2] == 0.5

    # Run expert to completion
    while not world.done:
        action = world.expert_action()
        obs, reward, done = world.step(action)

    assert world.visited_waypoint1
    assert world.success
    assert np.array_equal(world.position, world.waypoint2)


def test_waypoint_bonus_is_paid_once_and_hold_does_not_repeat_it():
    world = WaypointNavigationWorld(seed=77, delay=2, horizon=12)
    while not world.visited_waypoint1:
        _, reward, _ = world.step(world.expert_action())
    assert reward == 0.3
    for _ in range(2):
        observation, reward, done = world.step(6)
        assert reward == -0.01
        assert not done
        assert observation[world.dir2] == 0.25
    while not world.done:
        _, reward, _ = world.step(world.expert_action())
    assert world.success
    assert reward == 1.0


@pytest.mark.parametrize('amplitude', [True, False, '0.75', None, float('nan'),
                                       float('inf'), -0.1, 2.1])
def test_distractor_amplitude_rejects_invalid_values(amplitude):
    with pytest.raises(ValueError, match='distractor_amplitude'):
        DistractionNavigationWorld(seed=1, distractor_amplitude=amplitude)


def test_stress_expert_batch_conformance():
    for family in ('straight', 'distraction', 'waypoint'):
        batch = stress_expert_batch([101, 102], task_family=family, delay=2, horizon=8)
        assert batch['observations'].shape == (2, 8, 12)
        assert batch['actions'].shape == (2, 8)
        assert batch['mask'].shape == (2, 8)
        assert batch['executed_steps'] > 0
        assert batch['task_family'] == family
