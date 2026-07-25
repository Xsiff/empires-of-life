"""ASCII visualization for trained policy rollouts."""

from __future__ import annotations

import time

import torch

from eol.environment import Agent2D, CellType, Environment, RandomEnvironmentGenerator
from eol.micro.model import DirectionPolicy
from eol.micro.train import (
    TrainingConfig,
    encode_state,
    get_valid_action_mask,
    mask_action_logits,
    ACTION_SPACE,
)


def build_environment_frame(
    environment: Environment,
) -> list[list[CellType]]:
    """Build a renderable copy of the current environment grid."""

    return [row[:] for row in environment.grid]


def render_grid(grid: list[list[CellType]]) -> str:
    """Render a grid into an ASCII board."""

    symbols = {
        CellType.EMPTY: ".",
        CellType.OBSTACLE: "#",
        CellType.AGENT: "A",
        CellType.TARGET: "T",
    }
    return "\n".join(" ".join(symbols[cell] for cell in row) for row in grid)


def select_greedy_action(
    policy: DirectionPolicy,
    environment: Environment,
    agent: Agent2D,
) -> int:
    """Choose the highest-scoring action for visualization/evaluation."""

    state = encode_state(environment, agent).unsqueeze(0)
    valid_action_mask = get_valid_action_mask(environment, agent)
    with torch.no_grad():
        logits = mask_action_logits(policy(state), valid_action_mask)
    return int(torch.argmax(logits, dim=1).item())


def visualize_policy(
    policy: DirectionPolicy,
    environment_generator: RandomEnvironmentGenerator,
    max_steps: int,
    sleep_seconds: float = 0.25,
) -> bool:
    """Run one greedy episode and print the grid after each move."""

    environment, agents = environment_generator.generate_environment()
    agent = next(iter(agents))

    print("Initial environment:")
    print(render_grid(build_environment_frame(environment)))

    for step in range(1, max_steps + 1):
        if agent.position == environment.target_position:
            print(f"Target reached in {step - 1} steps.")
            return True

        action_index = select_greedy_action(policy=policy, environment=environment, agent=agent)
        action = ACTION_SPACE[action_index]
        environment.move_agent((agent, action))

        print("")
        print(f"Step {step}: action={action.name.lower()}")
        print(render_grid(build_environment_frame(environment)))

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if agent.position == environment.target_position:
            print(f"Target reached in {step} steps.")
            return True

    print("Target was not reached within the demo step limit.")
    return False


def visualize_multiple_environments(
    policy: DirectionPolicy,
    config: TrainingConfig,
    *,
    episodes: int | None = None,
    sleep_seconds: float | None = None,
) -> list[bool]:
    """Show greedy rollouts for several post-training environments."""

    results: list[bool] = []
    environment_count = (
        config.visualization_episodes if episodes is None else episodes
    )
    delay = (
        config.visualization_sleep_seconds
        if sleep_seconds is None
        else sleep_seconds
    )

    for environment_index in range(environment_count):
        demo_generator = RandomEnvironmentGenerator(
            size=config.grid_size,
            obstacle_count=config.obstacle_count,
            seed=config.seed + 1 + environment_index,
        )
        print("")
        print(
            f"=== Demo Environment {environment_index + 1}/"
            f"{environment_count} ==="
        )
        reached_target = visualize_policy(
            policy=policy,
            environment_generator=demo_generator,
            max_steps=config.max_steps,
            sleep_seconds=delay,
        )
        results.append(reached_target)

    successes = sum(results)
    print("")
    print(
        f"Demo summary: reached target in {successes}/"
        f"{environment_count} environments."
    )
    return results
