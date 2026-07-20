"""CLI for training and visualizing the micro policy."""

from __future__ import annotations

import argparse

from eol.micro.reward import RewardConfig
from eol.micro.train import TrainingConfig, train_policy
from eol.visualization import visualize_multiple_environments


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="eol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Run the training pipeline and visualize greedy rollouts.",
    )
    train_parser.add_argument("--grid-size", type=int, default=TrainingConfig.grid_size)
    train_parser.add_argument(
        "--obstacle-count",
        type=int,
        default=TrainingConfig.obstacle_count,
    )
    train_parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    train_parser.add_argument("--max-steps", type=int, default=TrainingConfig.max_steps)
    train_parser.add_argument(
        "--hidden-dim", type=int, default=TrainingConfig.hidden_dim
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=TrainingConfig.learning_rate,
    )
    train_parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    train_parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    train_parser.add_argument(
        "--print-every",
        type=int,
        default=TrainingConfig.print_every,
    )
    train_parser.add_argument(
        "--demo-sleep-seconds",
        type=float,
        default=TrainingConfig.demo_sleep_seconds,
    )
    train_parser.add_argument(
        "--demo-environment-count",
        type=int,
        default=TrainingConfig.demo_environment_count,
    )
    return parser


def run_training_pipeline(args: argparse.Namespace) -> None:
    """Train the policy and run post-training visualization demos."""

    training_config = TrainingConfig(
        grid_size=args.grid_size,
        obstacle_count=args.obstacle_count,
        episodes=args.episodes,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        print_every=args.print_every,
        demo_sleep_seconds=args.demo_sleep_seconds,
        demo_environment_count=args.demo_environment_count,
    )
    policy = train_policy(config=training_config, reward_config=RewardConfig())
    visualize_multiple_environments(policy=policy, config=training_config)


def main() -> None:
    """Execute the CLI."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_training_pipeline(args)


if __name__ == "__main__":
    main()
