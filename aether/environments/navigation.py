"""Delayed-goal 3D navigation with an initial cue and an actual action loop.

The expert has privileged goal access only while generating support trajectories.
The learned policy receives observe(), which never exposes the goal after tick 0.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np


MOVES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                  [0, 0, 1], [0, 0, -1], [0, 0, 0]], dtype=np.int64)


class NavigationWorld:
    def __init__(self, seed: int, delay: int = 3, horizon: int = 9,
                 distance: int = 2, radius: int = 3):
        values = (seed, delay, horizon, distance, radius)
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
            raise ValueError('world parameters must be integers')
        if seed < 0 or delay < 1 or distance < 1 or radius < distance or horizon < delay + distance:
            raise ValueError('require seed>=0, delay>=1, radius>=distance>=1 and horizon>=delay+distance')
        self.seed, self.delay, self.horizon = int(seed), int(delay), int(horizon)
        self.distance, self.radius = int(distance), int(radius)
        self.direction = int(np.random.default_rng(seed).integers(6))
        self.goal = MOVES[self.direction] * distance
        self.position = np.zeros(3, dtype=np.int64)
        self.timestep = 0
        self.success = False
        self.done = False

    def observe(self) -> np.ndarray:
        observation = np.zeros(12, dtype=np.float32)
        if self.timestep == 0:
            observation[self.direction] = 1
        observation[6:9] = self.position / self.radius
        observation[9] = max(0, self.delay - self.timestep) / self.horizon
        observation[10] = float(self.timestep >= self.delay)
        observation[11] = 1
        return observation

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < 7:
            raise ValueError('action must be an integer in [0, 6]')
        if self.done:
            raise ValueError('cannot step a terminated world')
        if self.timestep >= self.delay:
            self.position = np.clip(self.position + MOVES[action], -self.radius, self.radius)
        self.timestep += 1
        self.success = bool(np.array_equal(self.position, self.goal))
        self.done = self.success or self.timestep >= self.horizon
        reward = 1.0 if self.success else -0.01
        return self.observe(), reward, self.done

    def expert_action(self) -> int:
        if self.timestep < self.delay:
            return 6
        difference = self.goal - self.position
        for axis in range(3):
            if difference[axis]:
                return axis * 2 + int(difference[axis] < 0)
        return 6

    def clone(self) -> 'NavigationWorld':
        return deepcopy(self)


def expert_batch(seeds, delay: int = 3, horizon: int = 9) -> dict:
    seeds = list(seeds)
    if not seeds:
        raise ValueError('at least one support episode is required')
    observations = np.zeros((len(seeds), horizon, 12), dtype=np.float32)
    actions = np.full((len(seeds), horizon), 6, dtype=np.int64)
    mask = np.zeros((len(seeds), horizon), dtype=bool)
    executed = 0
    for row, seed in enumerate(seeds):
        world = NavigationWorld(seed=int(seed), delay=delay, horizon=horizon)
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
            'executed_steps': executed}
