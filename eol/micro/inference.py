"""Inference and evaluation helpers for micro navigation policies."""

from __future__ import annotations

import torch

from eol.environment import Agent2D, Environment
from eol.micro.config import NavigationConfig
from eol.micro.features import (
    ACTION_SPACE,
    encode_state,
    extract_policy_logits,
    get_valid_action_mask,
    mask_action_logits,
    resolve_action,
)
from eol.micro.scenario import ScenarioFactory, build_default_scenario_factory


def select_greedy_action(
    policy: torch.nn.Module,
    environment: Environment,
    agent: Agent2D,
) -> int:
    """Choose the highest-scoring action, halting if it cannot be executed."""

    state = encode_state(environment, agent).unsqueeze(0)
    with torch.no_grad():
        policy_output = policy(state)
        logits = extract_policy_logits(policy_output)
    if isinstance(policy_output, tuple):
        valid_action_mask = get_valid_action_mask(environment, agent)
        logits = mask_action_logits(logits, valid_action_mask)
    action_index = int(torch.argmax(logits, dim=1).item())
    resolved_action = resolve_action(environment, agent, action_index)
    return ACTION_SPACE.index(resolved_action)


def evaluate_policy(
    policy: torch.nn.Module,
    config: NavigationConfig,
    *,
    episodes: int | None = None,
    scenario_factory: ScenarioFactory | None = None,
) -> list[bool]:
    """Run greedy evaluation episodes on fresh environments."""

    evaluation_count = config.evaluation_episodes if episodes is None else episodes
    create_scenario = scenario_factory or build_default_scenario_factory(config)
    results: list[bool] = []
    for episode_index in range(evaluation_count):
        environment, agent = create_scenario(
            episode_index,
            config.seed + 10_000 + episode_index,
        )

        for _ in range(config.max_steps):
            if agent.position == environment.target_position:
                results.append(True)
                break

            action_index = select_greedy_action(policy, environment, agent)
            environment.move_agent((agent, ACTION_SPACE[action_index]))
        else:
            results.append(agent.position == environment.target_position)

    return results
