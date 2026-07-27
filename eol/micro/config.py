"""Configuration objects for micro-scale training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STATE_DIM = 31
ACTION_DIM = 5


@dataclass(frozen=True)
class NavigationConfig:
    """Shared configuration for single-agent navigation tasks."""

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
class TrainingConfig(NavigationConfig):
    """Configuration for the REINFORCE trainer."""


@dataclass(frozen=True)
class PretrainingConfig:
    """Configuration for A* imitation pretraining."""

    pretraining_episodes: int = 0
    pretraining_epochs: int = 10
    pretraining_batch_size: int = 64
    pretraining_learning_rate: float = 1e-3


@dataclass(frozen=True)
class PPOTrainingConfig(NavigationConfig, PretrainingConfig):
    """Configuration for PPO training."""

    episodes: int = 250
    learning_rate: float = 3e-4
    print_every: int = 10
    save_path: Path = Path("eol/micro/policy_ppo.pt")
    rollout_episodes_per_update: int = 8
    ppo_epochs: int = 4
    minibatch_size: int = 64
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
