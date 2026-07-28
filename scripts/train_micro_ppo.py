"""Programmatic entrypoint for PPO micro training."""

from __future__ import annotations

from dataclasses import replace

from eol.micro import PPOTrainingConfig, evaluate_policy, train_ppo_policy
from eol.micro.curriculum import get_curriculum_final_stage
from eol.visualization import visualize_multiple_environments


def main() -> None:
    config = PPOTrainingConfig()
    policy = train_ppo_policy(config)

    evaluation_config = config
    if config.curriculum:
        final_stage = get_curriculum_final_stage()
        evaluation_config = replace(
            config,
            grid_size=final_stage.grid_size,
            obstacle_count=final_stage.obstacle_count,
        )

    results = evaluate_policy(policy, evaluation_config)
    print(
        "Evaluation summary: reached target in "
        f"{sum(results)}/{len(results)} environments."
    )

    if evaluation_config.visualization_episodes > 0:
        visualize_multiple_environments(
            policy=policy,
            config=evaluation_config,
            episodes=evaluation_config.visualization_episodes,
            sleep_seconds=evaluation_config.visualization_sleep_seconds,
        )


if __name__ == "__main__":
    main()
