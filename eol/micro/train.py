"""Training loop for a grid-navigation policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch import optim
from torch.distributions import Categorical

from eol.environment import RandomEnvironmentGenerator
from eol.micro.model import DirectionPolicy
from eol.micro.reward import RewardConfig, compute_reward


ACTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for REINFORCE training."""

    grid_size: int = 6
    obstacle_count: int = 4
    episodes: int = 1_000
    max_steps: int = 30
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    seed: int = 7
    print_every: int = 100
    save_path: Path = Path("eol/micro/policy.pt")


def encode_state(
    agent_position: tuple[int, int],
    target_position: tuple[int, int],
    grid_size: int,
) -> torch.Tensor:
    """Encode positions into normalized model inputs."""

    max_index = max(grid_size - 1, 1)
    agent_row, agent_col = agent_position
    target_row, target_col = target_position
    delta_row = target_row - agent_row
    delta_col = target_col - agent_col

    return torch.tensor(
        [
            agent_row / max_index,
            agent_col / max_index,
            target_row / max_index,
            target_col / max_index,
            delta_row / max_index,
            delta_col / max_index,
        ],
        dtype=torch.float32,
    )


def manhattan_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Return Manhattan distance between two grid positions."""

    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def apply_action(
    agent_position: tuple[int, int],
    action_index: int,
    grid_size: int,
    obstacle_positions: set[tuple[int, int]],
) -> tuple[tuple[int, int], bool, bool]:
    """Move the agent if possible and report collisions."""

    delta_row, delta_col = ACTIONS[action_index]
    candidate_position = (
        agent_position[0] + delta_row,
        agent_position[1] + delta_col,
    )

    row, col = candidate_position
    if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
        return agent_position, False, True

    if candidate_position in obstacle_positions:
        return agent_position, True, False

    return candidate_position, False, False


def rollout_episode(
    policy: DirectionPolicy,
    environment_generator: RandomEnvironmentGenerator,
    reward_config: RewardConfig,
    max_steps: int,
) -> tuple[list[torch.Tensor], list[float], bool]:
    """Sample one training episode from the current policy."""

    environment = environment_generator.generate_environment()
    obstacle_positions = set(environment.obstacle_positions)
    agent_position = environment.agent_position
    target_position = environment.target_position

    log_probabilities: list[torch.Tensor] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        state = encode_state(agent_position, target_position, environment.size).unsqueeze(0)
        logits = policy(state)
        distribution = Categorical(logits=logits)
        action = distribution.sample()

        previous_distance = manhattan_distance(agent_position, target_position)
        next_position, hit_obstacle, hit_wall = apply_action(
            agent_position=agent_position,
            action_index=int(action.item()),
            grid_size=environment.size,
            obstacle_positions=obstacle_positions,
        )
        new_distance = manhattan_distance(next_position, target_position)
        reached_target = next_position == target_position

        reward = compute_reward(
            previous_distance=previous_distance,
            new_distance=new_distance,
            reached_target=reached_target,
            hit_obstacle=hit_obstacle,
            hit_wall=hit_wall,
            config=reward_config,
        )

        log_probabilities.append(distribution.log_prob(action))
        rewards.append(reward)
        agent_position = next_position

        if reached_target:
            return log_probabilities, rewards, True

    return log_probabilities, rewards, False


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns for REINFORCE."""

    returns: list[float] = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    returns_tensor = torch.tensor(returns, dtype=torch.float32)

    if len(returns_tensor) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std(unbiased=False) + 1e-8
        )

    return returns_tensor


def train_policy(
    config: TrainingConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> DirectionPolicy:
    """Train a policy that moves the agent toward the target."""

    training_config = config or TrainingConfig()
    rewards = reward_config or RewardConfig()

    random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    policy = DirectionPolicy(hidden_dim=training_config.hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=training_config.learning_rate)
    environment_generator = RandomEnvironmentGenerator(
        size=training_config.grid_size,
        obstacle_count=training_config.obstacle_count,
        seed=training_config.seed,
    )

    for episode in range(1, training_config.episodes + 1):
        optimizer.zero_grad()

        log_probabilities, episode_rewards, reached_target = rollout_episode(
            policy=policy,
            environment_generator=environment_generator,
            reward_config=rewards,
            max_steps=training_config.max_steps,
        )
        returns = discounted_returns(episode_rewards, training_config.gamma)
        loss = -torch.stack(
            [log_prob * episode_return for log_prob, episode_return in zip(log_probabilities, returns)]
        ).sum()
        loss.backward()
        optimizer.step()

        if episode % training_config.print_every == 0:
            total_reward = sum(episode_rewards)
            print(
                f"episode={episode} "
                f"reward={total_reward:.2f} "
                f"reached_target={reached_target}"
            )

    training_config.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), training_config.save_path)
    return policy


if __name__ == "__main__":
    train_policy()
