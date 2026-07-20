"""Reward utilities for micro-level RL training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward weights for the navigation task."""

    step_penalty: float = -0.05
    distance_scale: float = 0.4
    target_reward: float = 10.0
    obstacle_penalty: float = -2.0
    wall_penalty: float = -1.5


def compute_reward(
    previous_distance: int,
    new_distance: int,
    reached_target: bool,
    hit_obstacle: bool,
    hit_wall: bool,
    config: RewardConfig | None = None,
) -> float:
    """Calculate the reward for a single environment transition."""

    reward_config = config or RewardConfig()
    reward = reward_config.step_penalty
    reward += (previous_distance - new_distance) * reward_config.distance_scale

    if hit_obstacle:
        reward += reward_config.obstacle_penalty
    if hit_wall:
        reward += reward_config.wall_penalty
    if reached_target:
        reward += reward_config.target_reward

    return reward
