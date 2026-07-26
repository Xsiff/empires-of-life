"""Inference and evaluation helpers for micro navigation policies."""

from __future__ import annotations

import torch

from eol.environment import Agent2D, Environment, RandomEnvironmentGenerator
from eol.micro.train import (
    ACTION_SPACE,
    TrainingConfig,
    encode_state,
    resolve_action,
)


def select_greedy_action(
    policy: torch.nn.Module,
    environment: Environment,
    agent: Agent2D,
) -> int:
    """Choose the highest-scoring action, halting if it cannot be executed."""

    state = encode_state(environment, agent).unsqueeze(0)
    with torch.no_grad():
        logits = policy(state)
    action_index = int(torch.argmax(logits, dim=1).item())
    resolved_action = resolve_action(environment, agent, action_index)
    return ACTION_SPACE.index(resolved_action)


def evaluate_policy(
    policy: torch.nn.Module,
    config: TrainingConfig,
    *,
    episodes: int | None = None,
) -> list[bool]:
    """Run greedy evaluation episodes on fresh environments."""

    evaluation_count = config.evaluation_episodes if episodes is None else episodes
    results: list[bool] = []
    for episode_index in range(evaluation_count):
        generator = RandomEnvironmentGenerator(
            size=config.grid_size,
            obstacle_count=config.obstacle_count,
            seed=config.seed + 10_000 + episode_index,
        )
        environment, agents = generator.generate_environment()
        agent = next(iter(agents))

        for _ in range(config.max_steps):
            if agent.position == environment.target_position:
                results.append(True)
                break

            action_index = select_greedy_action(policy, environment, agent)
            environment.move_agent((agent, ACTION_SPACE[action_index]))
        else:
            results.append(agent.position == environment.target_position)

    return results
