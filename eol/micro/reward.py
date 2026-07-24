"""Reward utilities for micro-level RL training."""

from __future__ import annotations

from typing import Callable

from eol.environment import Environment


ACTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

RewardFn = Callable[[Environment, int], float]


def calculate_position(state: Environment, action: int) -> tuple[int, int]:
    """Return the candidate position for the proposed action."""

    agent_x, agent_y = state.agent_position
    delta_x, delta_y = ACTIONS[action]
    return agent_x + delta_x, agent_y + delta_y


class Reward:
    """Composable reward with a registry of state/action reward components."""

    _registry: dict[str, RewardFn] = {}
    _coeffs: dict[str, float] = {}

    def __init__(self, **kwargs: float) -> None:
        self.coeffs: dict[str, float] = {}

        for key, value in kwargs.items():
            if key not in self._registry:
                raise KeyError(f"function {key} is not a registered function")
            self.coeffs[key] = value

    @classmethod
    def register(
        cls, name: str, coef: float
    ) -> Callable[[RewardFn], RewardFn]:
        """Register a reward component with its default coefficient."""

        def decorator(func: RewardFn) -> RewardFn:
            cls._registry[name] = func
            cls._coeffs[name] = coef
            return func

        return decorator

    def compute_reward(self, state: Environment, action: int) -> float:
        """Compute total reward for the given state and action."""

        reward = 0.0
        for name, func in self._registry.items():
            coeff = self.coeffs.get(name, self._coeffs[name])
            reward += func(state, action) * coeff
        return reward


@Reward.register(name="step_penalty", coef=-0.1)
def step_penalty(state: Environment, action: int) -> float:
    """Apply the base per-step penalty."""

    del state, action
    return 1.0


@Reward.register(name="distance_progress", coef=0.6)
def distance_progress(state: Environment, action: int) -> float:
    """Reward moves that reduce distance to the target."""

    next_x, next_y = calculate_position(state, action)
    hit_wall = next_x < 0 or next_x >= state.size or next_y < 0 or next_y >= state.size
    hit_obstacle = (
        not hit_wall and (next_x, next_y) in set(state.obstacle_positions)
    )
    resolved_position = (
        state.agent_position if hit_wall or hit_obstacle else (next_x, next_y)
    )
    previous_distance = abs(state.agent_position[0] - state.target_position[0]) + abs(
        state.agent_position[1] - state.target_position[1]
    )
    new_distance = abs(resolved_position[0] - state.target_position[0]) + abs(
        resolved_position[1] - state.target_position[1]
    )
    return float(previous_distance - new_distance)


@Reward.register(name="hit_obstacle", coef=-2.0)
def hit_obstacle(state: Environment, action: int) -> float:
    """Penalize moving into an obstacle."""

    next_x, next_y = calculate_position(state, action)
    hit_wall = next_x < 0 or next_x >= state.size or next_y < 0 or next_y >= state.size
    return float(
        not hit_wall and (next_x, next_y) in set(state.obstacle_positions)
    )


@Reward.register(name="hit_wall", coef=-1.5)
def hit_wall(state: Environment, action: int) -> float:
    """Penalize moving into a wall."""

    next_x, next_y = calculate_position(state, action)
    return float(next_x < 0 or next_x >= state.size or next_y < 0 or next_y >= state.size)


@Reward.register(name="stagnation", coef=-0.3)
def stagnation(state: Environment, action: int) -> float:
    """Penalize actions that make no progress without reaching the target."""

    next_x, next_y = calculate_position(state, action)
    hit_wall = next_x < 0 or next_x >= state.size or next_y < 0 or next_y >= state.size
    hit_obstacle = (
        not hit_wall and (next_x, next_y) in set(state.obstacle_positions)
    )
    resolved_position = (
        state.agent_position if hit_wall or hit_obstacle else (next_x, next_y)
    )
    previous_distance = abs(state.agent_position[0] - state.target_position[0]) + abs(
        state.agent_position[1] - state.target_position[1]
    )
    new_distance = abs(resolved_position[0] - state.target_position[0]) + abs(
        resolved_position[1] - state.target_position[1]
    )
    reached_target = resolved_position == state.target_position
    return float((not reached_target) and new_distance == previous_distance)


@Reward.register(name="revisit_position", coef=-0.75)
def revisit_position(state: Environment, action: int) -> float:
    """Penalize returning to a previously visited position."""

    next_x, next_y = calculate_position(state, action)
    hit_wall = next_x < 0 or next_x >= state.size or next_y < 0 or next_y >= state.size
    hit_obstacle = (
        not hit_wall and (next_x, next_y) in set(state.obstacle_positions)
    )
    resolved_position = (
        state.agent_position if hit_wall or hit_obstacle else (next_x, next_y)
    )
    visited_positions = getattr(state, "visited_positions", ())
    return float(resolved_position in set(visited_positions))


@Reward.register(
    name="near_target_without_finish",
    coef=-0.5,
)
def near_target_without_finish(state: Environment, action: int) -> float:
    """Penalize hovering next to the target without finishing."""

    next_x, next_y = calculate_position(state, action)
    hit_wall = next_x < 0 or next_x >= state.size or next_y < 0 or next_y >= state.size
    hit_obstacle = (
        not hit_wall and (next_x, next_y) in set(state.obstacle_positions)
    )
    resolved_position = (
        state.agent_position if hit_wall or hit_obstacle else (next_x, next_y)
    )
    reached_target = resolved_position == state.target_position
    new_distance = abs(resolved_position[0] - state.target_position[0]) + abs(
        resolved_position[1] - state.target_position[1]
    )
    return float((not reached_target) and new_distance == 1)


@Reward.register(name="reach_target", coef=25.0)
def reach_target(state: Environment, action: int) -> float:
    """Reward reaching the target."""

    next_x, next_y = calculate_position(state, action)
    hit_wall = next_x < 0 or next_x >= state.size or next_y < 0 or next_y >= state.size
    hit_obstacle = (
        not hit_wall and (next_x, next_y) in set(state.obstacle_positions)
    )
    resolved_position = (
        state.agent_position if hit_wall or hit_obstacle else (next_x, next_y)
    )
    return float(resolved_position == state.target_position)


@Reward.register(name="miss_target", coef=-10.0)
def miss_target(state: Environment, action: int) -> float:
    """Penalize ending the episode without reaching the target."""

    next_x, next_y = calculate_position(state, action)
    hit_wall = next_x < 0 or next_x >= state.size or next_y < 0 or next_y >= state.size
    hit_obstacle = (
        not hit_wall and (next_x, next_y) in set(state.obstacle_positions)
    )
    resolved_position = (
        state.agent_position if hit_wall or hit_obstacle else (next_x, next_y)
    )
    reached_target = resolved_position == state.target_position

    if reached_target:
        return 0.0

    explicit_flag = getattr(state, "episode_finished", None)
    if explicit_flag is not None:
        return float(bool(explicit_flag))

    step_index = getattr(state, "step_index", None)
    max_steps = getattr(state, "max_steps", None)
    if step_index is not None and max_steps is not None:
        return float(step_index >= max_steps - 1)

    return 0.0


def compute_reward(
    state: Environment,
    action: int,
) -> float:
    """Compatibility wrapper around the registry-based Reward class."""

    reward = Reward()
    return reward.compute_reward(state, action)
