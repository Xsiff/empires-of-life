"""Reward utilities for micro-level RL training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward weights for the navigation task."""

    step_penalty: float = -0.1
    distance_scale: float = 0.6
    target_reward: float = 25.0
    missed_target_penalty: float = -10.0
    obstacle_penalty: float = -2.0
    wall_penalty: float = -1.5
    stagnation_penalty: float = -0.3
    revisit_penalty: float = -0.75
    near_target_without_finish_penalty: float = -0.5


def compute_reward(
    previous_distance: int,
    new_distance: int,
    reached_target: bool,
    hit_obstacle: bool,
    hit_wall: bool,
    revisited_position: bool,
    episode_finished: bool,
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
    if not reached_target and new_distance == previous_distance:
        reward += reward_config.stagnation_penalty
    if revisited_position:
        reward += reward_config.revisit_penalty
    if not reached_target and new_distance == 1:
        reward += reward_config.near_target_without_finish_penalty
    if reached_target:
        reward += reward_config.target_reward
    elif episode_finished:
        reward += reward_config.missed_target_penalty

    return reward
