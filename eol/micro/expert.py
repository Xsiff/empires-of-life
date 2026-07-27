"""Expert trajectory generation and imitation pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, optim

from eol.environment import Action, Environment
from eol.micro.config import PPOTrainingConfig
from eol.micro.features import (
    ACTION_SPACE,
    encode_state,
    get_valid_action_mask,
    manhattan_distance,
    mask_action_logits,
)
from eol.micro.model import ActorCriticPolicy
from eol.micro.scenario import ScenarioFactory, build_default_scenario_factory


@dataclass(frozen=True)
class ExpertSample:
    """One expert imitation sample from A* supervision."""

    state: torch.Tensor
    valid_action_mask: torch.Tensor
    action_index: int


def action_from_positions(current: tuple[int, int], nxt: tuple[int, int]) -> Action:
    """Convert two adjacent positions into an action."""

    delta = (nxt[0] - current[0], nxt[1] - current[1])
    for action in ACTION_SPACE:
        if action.value == delta:
            return action
    raise ValueError("Positions do not map to a supported action.")


def a_star_path(
    environment: Environment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Find a shortest path with A* using the current obstacle layout."""

    from heapq import heappop, heappush

    frontier: list[tuple[int, tuple[int, int]]] = []
    heappush(frontier, (0, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], int] = {start: 0}

    while frontier:
        _, current = heappop(frontier)
        if current == goal:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        for action in ACTION_SPACE[:-1]:
            neighbor = (current[0] + action.value[0], current[1] + action.value[1])
            if (
                not environment.in_bounds(neighbor)
                or neighbor in environment.obstacle_set
            ):
                continue

            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan_distance(neighbor, goal)
                heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    return None


def collect_expert_samples(
    config: PPOTrainingConfig,
    *,
    scenario_factory: ScenarioFactory | None = None,
) -> tuple[ExpertSample, ...]:
    """Collect A*-generated expert state-action supervision samples."""

    create_scenario = scenario_factory or build_default_scenario_factory(config)
    samples: list[ExpertSample] = []

    for episode_index in range(config.pretraining_episodes):
        environment, agent = create_scenario(
            episode_index,
            config.seed + 50_000 + episode_index,
        )
        path = a_star_path(
            environment,
            start=agent.position,
            goal=environment.target_position,
        )
        if path is None or len(path) < 2:
            continue

        for current_position, next_position in zip(path, path[1:]):
            state = encode_state(environment, agent)
            valid_action_mask = get_valid_action_mask(environment, agent)
            expert_action = action_from_positions(current_position, next_position)
            action_index = ACTION_SPACE.index(expert_action)
            samples.append(
                ExpertSample(
                    state=state,
                    valid_action_mask=valid_action_mask,
                    action_index=action_index,
                )
            )
            environment.move_agent((agent, expert_action))

    return tuple(samples)


def pretrain_policy_with_astar(
    policy: ActorCriticPolicy,
    config: PPOTrainingConfig,
    *,
    scenario_factory: ScenarioFactory | None = None,
) -> None:
    """Warm-start the PPO actor with A*-generated expert trajectories."""

    if config.pretraining_episodes <= 0:
        return

    samples = collect_expert_samples(config, scenario_factory=scenario_factory)
    if not samples:
        return

    optimizer = optim.Adam(
        policy.parameters(),
        lr=config.pretraining_learning_rate,
    )
    states = torch.stack([sample.state for sample in samples])
    valid_action_masks = torch.stack([sample.valid_action_mask for sample in samples])
    action_indices = torch.tensor(
        [sample.action_index for sample in samples],
        dtype=torch.int64,
    )
    sample_count = len(samples)
    indices = torch.arange(sample_count)

    for _ in range(config.pretraining_epochs):
        permutation = indices[torch.randperm(sample_count)]
        for minibatch_start in range(0, sample_count, config.pretraining_batch_size):
            minibatch_indices = permutation[
                minibatch_start : minibatch_start + config.pretraining_batch_size
            ]
            logits, _ = policy(states[minibatch_indices])
            masked_logits = mask_action_logits(
                logits,
                valid_action_masks[minibatch_indices],
            )
            loss = torch.nn.functional.cross_entropy(
                masked_logits,
                action_indices[minibatch_indices],
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
