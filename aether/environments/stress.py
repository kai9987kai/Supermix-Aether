"""Advanced communication-demanding task families: sensory distractors and multi-waypoint routes.

These environments adhere strictly to the 12-dimensional observation and 7-action
discrete contract, preserving exact deterministic reproducibility per seed.
"""

from copy import deepcopy
from numbers import Integral
import numpy as np

from aether.environments.navigation import MOVES, NavigationWorld


TASK_FAMILIES = ('straight', 'distraction', 'waypoint')


class DistractionNavigationWorld(NavigationWorld):
    """3D Navigation with intermediate sensory distractor pulses during the waiting delay.

    Tests whether frequency, phase, or attention-gated communication can filter
    false intermediate cues from contaminating memory across regions.
    """

    def __init__(self, seed: int, delay: int = 3, horizon: int = 9,
                 distance: int = 2, radius: int = 3,
                 distractor_amplitude: float = 0.75):
        super().__init__(seed=seed, delay=delay, horizon=horizon, distance=distance, radius=radius)
        if not 0.0 <= distractor_amplitude <= 2.0:
            raise ValueError('distractor_amplitude must be in [0, 2]')
        self.distractor_amplitude = float(distractor_amplitude)
        # Deterministic distractor schedule per seed
        rng = np.random.default_rng(seed + 54321)
        # Choose distinct wrong directions for intermediate ticks
        available_distractors = [d for d in range(6) if d != self.direction]
        self.distractor_dirs = rng.choice(available_distractors, size=max(1, delay), replace=True)

    def observe(self) -> np.ndarray:
        observation = np.zeros(12, dtype=np.float32)
        if self.timestep == 0:
            # True cue
            observation[self.direction] = 1.0
        elif 0 < self.timestep < self.delay:
            # Distractor pulse on wrong direction channel
            distractor_dir = int(self.distractor_dirs[self.timestep % len(self.distractor_dirs)])
            observation[distractor_dir] = self.distractor_amplitude

        observation[6:9] = self.position / self.radius
        observation[9] = max(0, self.delay - self.timestep) / self.horizon
        observation[10] = float(self.timestep >= self.delay)
        observation[11] = 1.0
        return observation


class WaypointNavigationWorld:
    """Sequential 3D waypoint navigation.

    Requires visiting waypoint 1 before navigating to waypoint 2.
    Tests multi-mode concurrent carrier memory.
    """

    def __init__(self, seed: int, delay: int = 3, horizon: int = 12,
                 distance: int = 2, radius: int = 4):
        values = (seed, delay, horizon, distance, radius)
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
            raise ValueError('world parameters must be integers')
        if seed < 0 or delay < 1 or distance < 1 or radius < distance or horizon < delay + 2 * distance:
            raise ValueError('invalid waypoint world parameters')

        self.seed, self.delay, self.horizon = int(seed), int(delay), int(horizon)
        self.distance, self.radius = int(distance), int(radius)
        rng = np.random.default_rng(seed)

        dirs = rng.choice(6, size=2, replace=False)
        self.dir1, self.dir2 = int(dirs[0]), int(dirs[1])
        self.waypoint1 = MOVES[self.dir1] * distance
        # Waypoint 2 is relative turn from waypoint 1
        self.waypoint2 = np.clip(self.waypoint1 + MOVES[self.dir2] * distance, -radius, radius)

        self.position = np.zeros(3, dtype=np.int64)
        self.timestep = 0
        self.visited_waypoint1 = False
        self.success = False
        self.done = False

    @property
    def goal(self) -> np.ndarray:
        return self.waypoint2 if self.visited_waypoint1 else self.waypoint1

    def observe(self) -> np.ndarray:
        observation = np.zeros(12, dtype=np.float32)
        if self.timestep == 0:
            # Primary and secondary waypoints encoded in directional cue slots
            observation[self.dir1] = 1.0
            observation[self.dir2] = 0.5
        elif self.visited_waypoint1:
            # Confirm stage 1 reached
            observation[self.dir2] = 0.25

        observation[6:9] = self.position / self.radius
        observation[9] = max(0, self.delay - self.timestep) / self.horizon
        observation[10] = float(self.timestep >= self.delay)
        observation[11] = 1.0
        return observation

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < 7:
            raise ValueError('action must be an integer in [0, 6]')
        if self.done:
            raise ValueError('cannot step a terminated world')

        if self.timestep >= self.delay:
            self.position = np.clip(self.position + MOVES[action], -self.radius, self.radius)

        if not self.visited_waypoint1 and np.array_equal(self.position, self.waypoint1):
            self.visited_waypoint1 = True

        self.timestep += 1
        self.success = self.visited_waypoint1 and bool(np.array_equal(self.position, self.waypoint2))
        self.done = self.success or self.timestep >= self.horizon
        reward = 1.0 if self.success else (0.3 if self.visited_waypoint1 else -0.01)
        return self.observe(), reward, self.done

    def expert_action(self) -> int:
        if self.timestep < self.delay:
            return 6  # Hold
        target = self.waypoint2 if self.visited_waypoint1 else self.waypoint1
        difference = target - self.position
        for axis in range(3):
            if difference[axis]:
                return axis * 2 + int(difference[axis] < 0)
        return 6

    def clone(self) -> 'WaypointNavigationWorld':
        return deepcopy(self)


def create_world(task_family: str = 'straight', seed: int = 0, **kwargs):
    """Instantiate a world from a task family name."""
    if task_family == 'straight':
        return NavigationWorld(seed=seed, **kwargs)
    elif task_family == 'distraction':
        return DistractionNavigationWorld(seed=seed, **kwargs)
    elif task_family == 'waypoint':
        return WaypointNavigationWorld(seed=seed, **kwargs)
    else:
        raise ValueError(f'unknown task_family {task_family!r}, expected one of {TASK_FAMILIES}')


def stress_expert_batch(seeds, task_family: str = 'straight', delay: int = 3,
                        horizon: int = 9, **kwargs) -> dict:
    """Generate expert batches for any supported task family."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError('at least one support episode is required')
    observations = np.zeros((len(seeds), horizon, 12), dtype=np.float32)
    actions = np.full((len(seeds), horizon), 6, dtype=np.int64)
    mask = np.zeros((len(seeds), horizon), dtype=bool)
    executed = 0
    for row, seed in enumerate(seeds):
        world = create_world(task_family, seed=int(seed), delay=delay, horizon=horizon, **kwargs)
        for tick in range(horizon):
            observations[row, tick] = world.observe()
            if world.done:
                continue
            action = world.expert_action()
            actions[row, tick] = action
            mask[row, tick] = tick >= delay
            world.step(action)
            executed += 1
    return {'observations': observations, 'actions': actions, 'mask': mask,
            'executed_steps': executed, 'task_family': task_family}
