"""Scenario generation helpers for micro training."""

from __future__ import annotations

from collections.abc import Callable

from eol.environment import Agent2D, Environment, RandomEnvironmentGenerator
from eol.micro.config import NavigationConfig


ScenarioFactory = Callable[[int, int], tuple[Environment, Agent2D]]


def build_single_agent_scenario(
    config: NavigationConfig,
    *,
    seed: int,
) -> tuple[Environment, Agent2D]:
    """Generate one fresh single-agent navigation scenario."""

    generator = RandomEnvironmentGenerator(
        size=config.grid_size,
        obstacle_count=config.obstacle_count,
        seed=seed,
    )
    environment, agents = generator.generate_environment()
    return environment, next(iter(agents))


def build_default_scenario_factory(config: NavigationConfig) -> ScenarioFactory:
    """Return the default scenario factory for single-agent navigation."""

    def create_scenario(_: int, seed: int) -> tuple[Environment, Agent2D]:
        return build_single_agent_scenario(config, seed=seed)

    return create_scenario
