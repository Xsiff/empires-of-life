"""CLI for training and visualizing the micro policy."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from eol.micro.algorithms import train_policy, train_ppo_policy
from eol.micro.curriculum import get_curriculum_final_stage
from eol.micro.inference import evaluate_policy
from eol.micro.config import PPOTrainingConfig, TrainingConfig
from eol.visualization import visualize_multiple_environments


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="eol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Run the training pipeline and visualize greedy rollouts.",
    )
    train_parser.add_argument(
        "--algorithm",
        choices=("reinforce", "ppo"),
        default="reinforce",
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
        "--evaluation-episodes",
        type=int,
        default=TrainingConfig.evaluation_episodes,
    )
    train_parser.add_argument(
        "--visualization-episodes",
        type=int,
        default=TrainingConfig.visualization_episodes,
    )
    train_parser.add_argument(
        "--visualization-sleep-seconds",
        type=float,
        default=TrainingConfig.visualization_sleep_seconds,
    )
    train_parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train with the staged curriculum schedule.",
    )
    train_parser.add_argument(
        "--save-path", type=Path, default=TrainingConfig.save_path
    )
    train_parser.add_argument(
        "--rollout-episodes-per-update",
        type=int,
        default=PPOTrainingConfig.rollout_episodes_per_update,
    )
    train_parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=PPOTrainingConfig.ppo_epochs,
    )
    train_parser.add_argument(
        "--minibatch-size",
        type=int,
        default=PPOTrainingConfig.minibatch_size,
    )
    train_parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=PPOTrainingConfig.clip_epsilon,
    )
    train_parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=PPOTrainingConfig.value_loss_coef,
    )
    train_parser.add_argument(
        "--entropy-coef",
        type=float,
        default=PPOTrainingConfig.entropy_coef,
    )
    train_parser.add_argument(
        "--gae-lambda",
        type=float,
        default=PPOTrainingConfig.gae_lambda,
    )
    train_parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=PPOTrainingConfig.max_grad_norm,
    )
    train_parser.add_argument(
        "--pretraining-episodes",
        type=int,
        default=PPOTrainingConfig.pretraining_episodes,
    )
    train_parser.add_argument(
        "--pretraining-epochs",
        type=int,
        default=PPOTrainingConfig.pretraining_epochs,
    )
    train_parser.add_argument(
        "--pretraining-batch-size",
        type=int,
        default=PPOTrainingConfig.pretraining_batch_size,
    )
    train_parser.add_argument(
        "--pretraining-learning-rate",
        type=float,
        default=PPOTrainingConfig.pretraining_learning_rate,
    )
    return parser


def run_training_pipeline(args: argparse.Namespace) -> None:
    """Train the policy and run post-training visualization demos."""

    if args.algorithm == "ppo":
        training_config = PPOTrainingConfig(
            grid_size=args.grid_size,
            obstacle_count=args.obstacle_count,
            episodes=args.episodes,
            max_steps=args.max_steps,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            seed=args.seed,
            print_every=args.print_every,
            save_path=args.save_path,
            evaluation_episodes=args.evaluation_episodes,
            visualization_episodes=args.visualization_episodes,
            visualization_sleep_seconds=args.visualization_sleep_seconds,
            curriculum=args.curriculum,
            rollout_episodes_per_update=args.rollout_episodes_per_update,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_epsilon=args.clip_epsilon,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            gae_lambda=args.gae_lambda,
            max_grad_norm=args.max_grad_norm,
            pretraining_episodes=args.pretraining_episodes,
            pretraining_epochs=args.pretraining_epochs,
            pretraining_batch_size=args.pretraining_batch_size,
            pretraining_learning_rate=args.pretraining_learning_rate,
        )
        policy = train_ppo_policy(config=training_config)
    else:
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
            save_path=args.save_path,
            evaluation_episodes=args.evaluation_episodes,
            visualization_episodes=args.visualization_episodes,
            visualization_sleep_seconds=args.visualization_sleep_seconds,
            curriculum=args.curriculum,
        )
        policy = train_policy(config=training_config)
    evaluation_config = training_config
    if training_config.curriculum:
        final_stage = get_curriculum_final_stage()
        evaluation_config = replace(
            training_config,
            grid_size=final_stage.grid_size,
            obstacle_count=final_stage.obstacle_count,
        )
    results = evaluate_policy(policy=policy, config=evaluation_config)
    successes = sum(results)
    print(
        f"Evaluation summary: reached target in "
        f"{successes}/{len(results)} environments."
    )
    if training_config.visualization_episodes > 0:
        visualize_multiple_environments(
            policy=policy,
            config=evaluation_config,
            episodes=evaluation_config.visualization_episodes,
            sleep_seconds=evaluation_config.visualization_sleep_seconds,
        )


def main() -> None:
    """Execute the CLI."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_training_pipeline(args)


if __name__ == "__main__":
    main()
