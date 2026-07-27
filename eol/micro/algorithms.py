"""Training algorithms for micro navigation."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import nn, optim
from torch.distributions import Categorical

from eol.micro.config import ACTION_DIM, PPOTrainingConfig, STATE_DIM, TrainingConfig
from eol.micro.expert import pretrain_policy_with_astar
from eol.micro.features import mask_action_logits
from eol.micro.model import ActorCriticPolicy, DirectionPolicy
from eol.micro.rollout import (
    EpisodeTrajectory,
    PPOTrajectory,
    collect_episode,
    collect_ppo_episode,
)
from eol.micro.scenario import ScenarioFactory, build_default_scenario_factory


@dataclass(frozen=True)
class TrainMetrics:
    """Summary metrics for one training episode."""

    episode_index: int
    total_reward: float
    episode_length: int
    reached_target: bool


@dataclass(frozen=True)
class PPOBatch:
    """Flattened PPO rollout tensors."""

    states: torch.Tensor
    valid_action_masks: torch.Tensor
    action_indices: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def discounted_returns(trajectory: EpisodeTrajectory, gamma: float) -> torch.Tensor:
    """Compute discounted returns for the collected trajectory."""

    returns: list[float] = []
    running_return = 0.0
    for step in reversed(trajectory.steps):
        running_return = step.reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns for one trajectory."""

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.tensor(0.0, dtype=rewards.dtype)

    for index in range(len(rewards) - 1, -1, -1):
        next_value = (
            values[index + 1]
            if index + 1 < len(values)
            else torch.tensor(0.0, dtype=values.dtype)
        )
        non_terminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[index] = last_advantage

    returns = advantages + values
    return advantages, returns


def build_ppo_batch(
    trajectories: tuple[PPOTrajectory, ...],
    gamma: float,
    gae_lambda: float,
) -> PPOBatch:
    """Flatten PPO trajectories into training tensors."""

    state_tensors: list[torch.Tensor] = []
    mask_tensors: list[torch.Tensor] = []
    action_indices: list[int] = []
    old_log_probs: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    all_returns: list[torch.Tensor] = []
    all_advantages: list[torch.Tensor] = []

    for trajectory in trajectories:
        rewards = torch.tensor(
            [step.reward for step in trajectory.steps], dtype=torch.float32
        )
        values = torch.stack([step.value for step in trajectory.steps]).to(
            torch.float32
        )
        dones = torch.tensor(
            [1.0 if step.done else 0.0 for step in trajectory.steps],
            dtype=torch.float32,
        )
        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

        state_tensors.extend(step.state for step in trajectory.steps)
        mask_tensors.extend(step.valid_action_mask for step in trajectory.steps)
        action_indices.extend(step.action_index for step in trajectory.steps)
        old_log_probs.extend(step.old_log_prob for step in trajectory.steps)
        old_values.extend(step.value for step in trajectory.steps)
        all_advantages.extend(advantages)
        all_returns.extend(returns)

    assert state_tensors
    advantages_tensor = torch.stack(all_advantages)
    if len(advantages_tensor) > 1:
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std(unbiased=False) + 1e-8
        )

    return PPOBatch(
        states=torch.stack(state_tensors),
        valid_action_masks=torch.stack(mask_tensors),
        action_indices=torch.tensor(action_indices, dtype=torch.int64),
        old_log_probs=torch.stack(old_log_probs).to(torch.float32),
        old_values=torch.stack(old_values).to(torch.float32),
        returns=torch.stack(all_returns).to(torch.float32),
        advantages=advantages_tensor.to(torch.float32),
    )


def ppo_losses(
    policy: ActorCriticPolicy,
    batch: PPOBatch,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO total, policy, value, and entropy losses."""

    logits, values = policy(batch.states)
    masked_logits = mask_action_logits(logits, batch.valid_action_masks)
    distribution = Categorical(logits=masked_logits)
    new_log_probs = distribution.log_prob(batch.action_indices)
    entropy = distribution.entropy().mean()
    ratios = torch.exp(new_log_probs - batch.old_log_probs)
    unclipped = ratios * batch.advantages
    clipped = (
        torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch.advantages
    )
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = torch.nn.functional.mse_loss(values, batch.returns)
    total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
    return total_loss, policy_loss, value_loss, entropy


def _build_metrics(episode_index: int, trajectory: EpisodeTrajectory) -> TrainMetrics:
    return TrainMetrics(
        episode_index=episode_index,
        total_reward=trajectory.total_reward,
        episode_length=len(trajectory.steps),
        reached_target=trajectory.reached_target,
    )


def train_policy(
    config: TrainingConfig,
    *,
    scenario_factory: ScenarioFactory | None = None,
) -> DirectionPolicy:
    """Train a single-agent navigation policy with REINFORCE."""

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    create_scenario = scenario_factory or build_default_scenario_factory(config)
    policy = DirectionPolicy(
        input_dim=STATE_DIM,
        hidden_dim=config.hidden_dim,
        action_dim=ACTION_DIM,
    )
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
    metrics: list[TrainMetrics] = []

    for episode_index in range(1, config.episodes + 1):
        environment, agent = create_scenario(episode_index, config.seed + episode_index)
        trajectory = collect_episode(
            policy,
            environment,
            agent,
            max_steps=config.max_steps,
        )
        returns = discounted_returns(trajectory, config.gamma)

        if trajectory.steps:
            log_probs = torch.stack([step.log_prob for step in trajectory.steps])
            advantages = returns
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )
            policy_loss = -(log_probs * advantages).sum()
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        metrics.append(_build_metrics(episode_index, trajectory))
        if config.print_every > 0 and episode_index % config.print_every == 0:
            recent = metrics[-config.print_every :]
            avg_reward = sum(metric.total_reward for metric in recent) / len(recent)
            successes = sum(metric.reached_target for metric in recent)
            print(
                f"episode={episode_index} avg_reward={avg_reward:.3f} "
                f"successes={successes}/{len(recent)}"
            )

    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), config.save_path)
    return policy


def train_ppo_policy(
    config: PPOTrainingConfig,
    *,
    scenario_factory: ScenarioFactory | None = None,
) -> ActorCriticPolicy:
    """Train a single-agent navigation policy with PPO."""

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    create_scenario = scenario_factory or build_default_scenario_factory(config)
    policy = ActorCriticPolicy(
        input_dim=STATE_DIM,
        hidden_dim=config.hidden_dim,
        action_dim=ACTION_DIM,
    )
    pretrain_policy_with_astar(policy, config, scenario_factory=create_scenario)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
    metrics: list[TrainMetrics] = []

    for update_index in range(1, config.episodes + 1):
        trajectories: list[PPOTrajectory] = []
        for rollout_index in range(config.rollout_episodes_per_update):
            seed = (
                config.seed
                + update_index * config.rollout_episodes_per_update
                + rollout_index
            )
            environment, agent = create_scenario(rollout_index, seed)
            trajectory = collect_ppo_episode(
                policy,
                environment,
                agent,
                max_steps=config.max_steps,
            )
            trajectories.append(trajectory)
            metrics.append(
                TrainMetrics(
                    episode_index=len(metrics) + 1,
                    total_reward=trajectory.total_reward,
                    episode_length=len(trajectory.steps),
                    reached_target=trajectory.reached_target,
                )
            )

        batch = build_ppo_batch(tuple(trajectories), config.gamma, config.gae_lambda)
        sample_count = batch.states.shape[0]
        indices = torch.arange(sample_count)

        for _ in range(config.ppo_epochs):
            permutation = indices[torch.randperm(sample_count)]
            for minibatch_start in range(0, sample_count, config.minibatch_size):
                minibatch_indices = permutation[
                    minibatch_start : minibatch_start + config.minibatch_size
                ]
                minibatch = PPOBatch(
                    states=batch.states[minibatch_indices],
                    valid_action_masks=batch.valid_action_masks[minibatch_indices],
                    action_indices=batch.action_indices[minibatch_indices],
                    old_log_probs=batch.old_log_probs[minibatch_indices],
                    old_values=batch.old_values[minibatch_indices],
                    returns=batch.returns[minibatch_indices],
                    advantages=batch.advantages[minibatch_indices],
                )
                total_loss, _, _, _ = ppo_losses(
                    policy,
                    minibatch,
                    config.clip_epsilon,
                    config.value_loss_coef,
                    config.entropy_coef,
                )
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

        if config.print_every > 0 and update_index % config.print_every == 0:
            recent = metrics[-config.print_every * config.rollout_episodes_per_update :]
            avg_reward = sum(metric.total_reward for metric in recent) / len(recent)
            successes = sum(metric.reached_target for metric in recent)
            print(
                f"update={update_index} avg_reward={avg_reward:.3f} "
                f"successes={successes}/{len(recent)}"
            )

    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), config.save_path)
    return policy
