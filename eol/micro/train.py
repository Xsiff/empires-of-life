"""Single-agent navigation training pipeline for micro control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch import optim
from torch.distributions import Categorical

from eol.environment import Action, Agent2D, Environment, RandomEnvironmentGenerator
from eol.micro.model import DirectionPolicy
from eol.micro.reward import compute_reward


ACTION_SPACE: tuple[Action, ...] = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for micro-level training runs."""

    grid_size: int = 8
    obstacle_count: int = 6
    episodes: int = 1_000
    max_steps: int = 32
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    seed: int = 7
    print_every: int = 50
    save_path: Path = Path("eol/micro/policy.pt")
    evaluation_episodes: int = 5
    visualization_episodes: int = 3
    visualization_sleep_seconds: float = 0.15


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
class TrainMetrics:
    """Summary metrics for one training episode."""

    episode_index: int
    total_reward: float
    episode_length: int
    reached_target: bool


def encode_state(environment: Environment, agent: Agent2D) -> torch.Tensor:
    """Encode one agent's navigation state as a fixed-size tensor."""

    agent_row, agent_col = agent.position
    target_row, target_col = environment.target_position
    delta_row = target_row - agent_row
    delta_col = target_col - agent_col

    height_scale = max(environment.height - 1, 1)
    width_scale = max(environment.width - 1, 1)

    blocked_features = []
    for action in ACTION_SPACE:
        blocked_features.append(0.0 if environment.can_move_to((agent, action)) else 1.0)

    return torch.tensor(
        [
            agent_row / height_scale,
            agent_col / width_scale,
            target_row / height_scale,
            target_col / width_scale,
            delta_row / height_scale,
            delta_col / width_scale,
            *blocked_features,
        ],
        dtype=torch.float32,
    )


def get_valid_action_mask(environment: Environment, agent: Agent2D) -> torch.Tensor:
    """Return a binary mask for valid cardinal actions."""

    return torch.tensor(
        [
            1.0 if environment.can_move_to((agent, action)) else 0.0
            for action in ACTION_SPACE
        ],
        dtype=torch.float32,
    )


def mask_action_logits(
    logits: torch.Tensor, valid_action_mask: torch.Tensor
) -> torch.Tensor:
    """Mask invalid actions so policy selection ignores them."""

    invalid_mask = valid_action_mask.unsqueeze(0) <= 0
    masked_logits = logits.clone()
    return masked_logits.masked_fill(invalid_mask, torch.finfo(logits.dtype).min)


def select_action(
    policy: torch.nn.Module,
    environment: Environment,
    agent: Agent2D,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one valid action from the policy for the given agent."""

    state = encode_state(environment, agent)
    valid_action_mask = get_valid_action_mask(environment, agent)
    logits = policy(state.unsqueeze(0))
    masked_logits = mask_action_logits(logits, valid_action_mask)
    distribution = Categorical(logits=masked_logits)
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
        action = ACTION_SPACE[action_index]
        position_before = agent.position
        environment.move_agent((agent, action))
        reached_target = agent.position == environment.target_position
        episode_finished = reached_target or step_index == max_steps - 1

        reward = compute_reward(
            environment,
            agent,
            action_index,
            position_before=position_before,
            reached_target=reached_target,
            episode_finished=episode_finished,
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
                done=episode_finished,
                position_before=position_before,
                position_after=agent.position,
            )
        )

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


def discounted_returns(
    trajectory: EpisodeTrajectory, gamma: float
) -> torch.Tensor:
    """Compute discounted returns for the collected trajectory."""

    returns: list[float] = []
    running_return = 0.0
    for step in reversed(trajectory.steps):
        running_return = step.reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


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
        environment, (agent, ) = generator.generate_environment()

        for _ in range(config.max_steps):
            if agent.position == environment.target_position:
                results.append(True)
                break

            state = encode_state(environment, agent).unsqueeze(0)
            valid_action_mask = get_valid_action_mask(environment, agent)
            with torch.no_grad():
                logits = mask_action_logits(policy(state), valid_action_mask)
            action_index = int(torch.argmax(logits, dim=1).item())
            environment.move_agent((agent, ACTION_SPACE[action_index]))
        else:
            results.append(agent.position == environment.target_position)

    return results


def _build_metrics(
    episode_index: int, trajectory: EpisodeTrajectory
) -> TrainMetrics:
    return TrainMetrics(
        episode_index=episode_index,
        total_reward=trajectory.total_reward,
        episode_length=len(trajectory.steps),
        reached_target=trajectory.reached_target,
    )


def train_policy(config: TrainingConfig) -> torch.nn.Module:
    """Train a single-agent navigation policy with a simple policy gradient."""

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    policy = DirectionPolicy(input_dim=10, hidden_dim=config.hidden_dim, action_dim=4)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

    metrics: list[TrainMetrics] = []

    for episode_index in range(1, config.episodes + 1):
        generator = RandomEnvironmentGenerator(
            size=config.grid_size,
            obstacle_count=config.obstacle_count,
            seed=config.seed + episode_index,
        )
        environment, agents = generator.generate_environment()
        agent = next(iter(agents))

        trajectory = collect_episode(
            policy,
            environment,
            agent,
            max_steps=config.max_steps,
        )
        returns = discounted_returns(trajectory, config.gamma)

        if len(trajectory.steps) > 0:
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
