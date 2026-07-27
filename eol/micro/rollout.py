"""Rollout collection for micro navigation policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from eol.environment import Action, Agent2D, Environment
from eol.micro.features import (
    encode_state,
    extract_policy_logits,
    get_valid_action_mask,
    mask_action_logits,
    resolve_action,
)
from eol.micro.model import ActorCriticPolicy
from eol.micro.reward import compute_reward


@dataclass(frozen=True)
class EpisodeStep:
    """One recorded interaction step from an episode rollout."""

    state: torch.Tensor
    valid_action_mask: torch.Tensor
    action_index: int
    action: Action
    log_prob: torch.Tensor
    reward: float
    done: bool
    position_before: tuple[int, int]
    position_after: tuple[int, int]


@dataclass(frozen=True)
class EpisodeTrajectory:
    """A collected single-agent episode trajectory."""

    steps: tuple[EpisodeStep, ...]
    total_reward: float
    reached_target: bool
    termination_reason: str


@dataclass(frozen=True)
class PPOStep:
    """One PPO rollout step."""

    state: torch.Tensor
    valid_action_mask: torch.Tensor
    action_index: int
    action: Action
    old_log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


@dataclass(frozen=True)
class PPOTrajectory:
    """One PPO rollout trajectory."""

    steps: tuple[PPOStep, ...]
    total_reward: float
    reached_target: bool
    termination_reason: str


def select_action(
    policy: torch.nn.Module,
    environment: Environment,
    agent: Agent2D,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one action from the policy for the given agent."""

    state = encode_state(environment, agent)
    valid_action_mask = get_valid_action_mask(environment, agent)
    logits = extract_policy_logits(policy(state.unsqueeze(0)))
    distribution = Categorical(logits=logits)
    action_index = int(distribution.sample().item())
    log_prob = distribution.log_prob(torch.tensor(action_index)).squeeze(0)
    return action_index, log_prob, state, valid_action_mask


def collect_episode(
    policy: torch.nn.Module,
    environment: Environment,
    agent: Agent2D,
    *,
    max_steps: int,
) -> EpisodeTrajectory:
    """Collect one rollout episode for a single navigation agent."""

    steps: list[EpisodeStep] = []
    total_reward = 0.0
    recent_positions: list[tuple[int, int]] = [agent.position]

    for step_index in range(max_steps):
        valid_action_mask = get_valid_action_mask(environment, agent)
        if torch.count_nonzero(valid_action_mask).item() == 0:
            return EpisodeTrajectory(
                steps=tuple(steps),
                total_reward=total_reward,
                reached_target=False,
                termination_reason="no_valid_actions",
            )

        action_index, log_prob, state, valid_action_mask = select_action(
            policy, environment, agent
        )
        action = resolve_action(environment, agent, action_index)
        position_before = agent.position
        environment.move_agent((agent, action))
        reached_target = agent.position == environment.target_position
        done = reached_target or step_index == max_steps - 1
        reward = compute_reward(
            environment,
            agent,
            action_index,
            position_before=position_before,
            reached_target=reached_target,
            episode_finished=done,
            recent_positions=tuple(recent_positions),
        )
        total_reward += reward
        steps.append(
            EpisodeStep(
                state=state,
                valid_action_mask=valid_action_mask,
                action_index=action_index,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                position_before=position_before,
                position_after=agent.position,
            )
        )
        recent_positions.append(agent.position)
        if reached_target:
            return EpisodeTrajectory(
                steps=tuple(steps),
                total_reward=total_reward,
                reached_target=True,
                termination_reason="target_reached",
            )

    return EpisodeTrajectory(
        steps=tuple(steps),
        total_reward=total_reward,
        reached_target=False,
        termination_reason="max_steps",
    )


def collect_ppo_episode(
    policy: ActorCriticPolicy,
    environment: Environment,
    agent: Agent2D,
    *,
    max_steps: int,
) -> PPOTrajectory:
    """Collect one PPO rollout episode for a single navigation agent."""

    steps: list[PPOStep] = []
    total_reward = 0.0
    recent_positions: list[tuple[int, int]] = [agent.position]

    for step_index in range(max_steps):
        state = encode_state(environment, agent)
        valid_action_mask = get_valid_action_mask(environment, agent)
        logits, value = policy(state.unsqueeze(0))
        masked_logits = mask_action_logits(logits, valid_action_mask)
        distribution = Categorical(logits=masked_logits)
        action_index = int(distribution.sample().item())
        old_log_prob = distribution.log_prob(torch.tensor(action_index)).squeeze(0)
        action = resolve_action(environment, agent, action_index)
        position_before = agent.position
        environment.move_agent((agent, action))
        reached_target = agent.position == environment.target_position
        done = reached_target or step_index == max_steps - 1
        reward = compute_reward(
            environment,
            agent,
            action_index,
            position_before=position_before,
            reached_target=reached_target,
            episode_finished=done,
            recent_positions=tuple(recent_positions),
        )
        total_reward += reward
        steps.append(
            PPOStep(
                state=state,
                valid_action_mask=valid_action_mask,
                action_index=action_index,
                action=action,
                old_log_prob=old_log_prob.detach(),
                value=value.squeeze(0).detach(),
                reward=reward,
                done=done,
            )
        )
        recent_positions.append(agent.position)
        if reached_target:
            return PPOTrajectory(
                steps=tuple(steps),
                total_reward=total_reward,
                reached_target=True,
                termination_reason="target_reached",
            )

    return PPOTrajectory(
        steps=tuple(steps),
        total_reward=total_reward,
        reached_target=False,
        termination_reason="max_steps",
    )
