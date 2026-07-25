"""Reward utilities for the rebuilt micro navigation trainer."""

from __future__ import annotations

from typing import Callable

from eol.environment import Agent2D, Environment


RewardFn = Callable[..., float]


class Reward:
    """Composable reward with a registry of environment-based components."""

    _registry: dict[str, RewardFn] = {}
    _coeffs: dict[str, float] = {}

    def __init__(self, **kwargs: float) -> None:
        self.coeffs: dict[str, float] = {}
        for key, value in kwargs.items():
            if key not in self._registry:
                raise KeyError(f"function {key} is not a registered function")
            self.coeffs[key] = value

    @classmethod
    def register(cls, name: str, coef: float) -> Callable[[RewardFn], RewardFn]:
        """Register a reward component with its default coefficient."""

        def decorator(func: RewardFn) -> RewardFn:
            cls._registry[name] = func
            cls._coeffs[name] = coef
            return func

        return decorator

    def compute_reward(
        self,
        environment: Environment,
        agent: Agent2D,
        action_index: int,
        *,
        position_before: tuple[int, int],
        reached_target: bool,
        episode_finished: bool,
    ) -> float:
        """Compute total reward for the current navigation transition."""

        total_reward = 0.0
        for name, func in self._registry.items():
            coeff = self.coeffs.get(name, self._coeffs[name])
            total_reward += coeff * func(
                environment,
                agent,
                action_index,
                position_before=position_before,
                reached_target=reached_target,
                episode_finished=episode_finished,
            )
        return total_reward


def _target_distance(
    position: tuple[int, int], target_position: tuple[int, int]
) -> int:
    return abs(position[0] - target_position[0]) + abs(position[1] - target_position[1])


@Reward.register(name="step_penalty", coef=-0.1)
def step_penalty(
    environment: Environment,
    agent: Agent2D,
    action_index: int,
    **kwargs: object,
) -> float:
    """Apply the base per-step penalty."""

    del environment, agent, action_index, kwargs
    return 1.0


@Reward.register(name="distance_progress", coef=0.8)
def distance_progress(
    environment: Environment,
    agent: Agent2D,
    action_index: int,
    *,
    position_before: tuple[int, int],
    **kwargs: object,
) -> float:
    """Reward reductions in Manhattan distance to the target."""

    del action_index, kwargs
    previous_distance = _target_distance(position_before, environment.target_position)
    current_distance = _target_distance(agent.position, environment.target_position)
    return float(previous_distance - current_distance)


@Reward.register(name="stagnation", coef=-0.25)
def stagnation(
    environment: Environment,
    agent: Agent2D,
    action_index: int,
    *,
    position_before: tuple[int, int],
    reached_target: bool,
    **kwargs: object,
) -> float:
    """Penalize a step that fails to change position or make progress."""

    del action_index, kwargs
    if reached_target:
        return 0.0

    previous_distance = _target_distance(position_before, environment.target_position)
    current_distance = _target_distance(agent.position, environment.target_position)
    no_progress = agent.position == position_before or current_distance >= previous_distance
    return float(no_progress)


@Reward.register(name="reach_target", coef=25.0)
def reach_target(
    environment: Environment,
    agent: Agent2D,
    action_index: int,
    *,
    reached_target: bool,
    **kwargs: object,
) -> float:
    """Reward reaching the target."""

    del environment, agent, action_index, kwargs
    return float(reached_target)


@Reward.register(name="miss_target", coef=-5.0)
def miss_target(
    environment: Environment,
    agent: Agent2D,
    action_index: int,
    *,
    reached_target: bool,
    episode_finished: bool,
    **kwargs: object,
) -> float:
    """Penalize ending the episode without reaching the target."""

    del environment, agent, action_index, kwargs
    return float(episode_finished and not reached_target)


def compute_reward(
    environment: Environment,
    agent: Agent2D,
    action_index: int,
    *,
    position_before: tuple[int, int],
    reached_target: bool,
    episode_finished: bool,
) -> float:
    """Compatibility wrapper around the registry-based reward class."""

    reward = Reward()
    return reward.compute_reward(
        environment,
        agent,
        action_index,
        position_before=position_before,
        reached_target=reached_target,
        episode_finished=episode_finished,
    )
