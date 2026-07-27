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
        **kwargs: object,
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
                **kwargs,
            )
        return total_reward


@Reward.register(name="reach_target", coef=10.0)
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


@Reward.register(name="miss_target", coef=-1.0)
def miss_target(
    environment: Environment,
    agent: Agent2D,
    action_index: int,
    *,
    reached_target: bool,
    **kwargs: object,
) -> float:
    """Penalize every transition that does not reach the target."""

    del environment, agent, action_index, kwargs
    return float(not reached_target)


def compute_reward(
    environment: Environment,
    agent: Agent2D,
    action_index: int,
    *,
    position_before: tuple[int, int],
    reached_target: bool,
    episode_finished: bool,
    **kwargs: object,
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
        **kwargs,
    )
