"""ASCII visualization for trained policy rollouts."""

from __future__ import annotations

import time

import torch

from eol.environment import CellType, Environment, RandomEnvironmentGenerator
from eol.micro.model import DirectionPolicy
from eol.micro.train import (
    TrainingConfig,
    apply_action,
    encode_state,
    get_valid_action_mask,
    mask_action_logits,
)


def build_environment_frame(
    environment: Environment,
    agent_position: tuple[int, int],
) -> list[list[CellType]]:
    """Build a renderable grid for the current agent position."""

    grid = [
        [CellType.EMPTY for _ in range(environment.size)]
        for _ in range(environment.size)
    ]
    target_row, target_col = environment.target_position
    grid[target_row][target_col] = CellType.TARGET

    for obstacle_row, obstacle_col in environment.obstacle_positions:
        grid[obstacle_row][obstacle_col] = CellType.OBSTACLE

    agent_row, agent_col = agent_position
    grid[agent_row][agent_col] = CellType.AGENT
    return grid


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
    agent_position: tuple[int, int],
    target_position: tuple[int, int],
    grid_size: int,
    obstacle_positions: set[tuple[int, int]],
    visited_positions: set[tuple[int, int]] | None = None,
) -> int:
    """Choose the highest-scoring action for visualization/evaluation."""

    state = encode_state(
        agent_position,
        target_position,
        grid_size,
        obstacle_positions,
    ).unsqueeze(0)
    valid_action_mask = get_valid_action_mask(
        agent_position=agent_position,
        grid_size=grid_size,
        obstacle_positions=obstacle_positions,
    )
    with torch.no_grad():
        logits = mask_action_logits(policy(state), valid_action_mask)
    action_scores = logits.squeeze(0).clone()

    if visited_positions:
        revisit_mask = torch.zeros_like(action_scores, dtype=torch.bool)
        valid_mask = valid_action_mask.to(dtype=torch.bool)

        for action_index, (delta_row, delta_col) in enumerate(
            ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            candidate_position = (
                agent_position[0] + delta_row,
                agent_position[1] + delta_col,
            )
            if candidate_position in visited_positions:
                revisit_mask[action_index] = True

        unvisited_valid_mask = valid_mask & ~revisit_mask
        if torch.any(unvisited_valid_mask):
            action_scores = action_scores.masked_fill(
                valid_mask & revisit_mask,
                torch.finfo(action_scores.dtype).min,
            )

    return int(torch.argmax(action_scores).item())


def visualize_policy(
    policy: DirectionPolicy,
    environment_generator: RandomEnvironmentGenerator,
    max_steps: int,
    sleep_seconds: float = 0.25,
) -> bool:
    """Run one greedy episode and print the grid after each move."""

    environment, _ = environment_generator.generate_environment()
    obstacle_positions = set(environment.obstacle_positions)
    agent_position = environment.agent_position
    visited_positions = {agent_position}

    print("Initial environment:")
    print(render_grid(build_environment_frame(environment, agent_position)))

    for step in range(1, max_steps + 1):
        if agent_position == environment.target_position:
            print(f"Target reached in {step - 1} steps.")
            return True

        action_index = select_greedy_action(
            policy=policy,
            agent_position=agent_position,
            target_position=environment.target_position,
            grid_size=environment.size,
            obstacle_positions=obstacle_positions,
            visited_positions=visited_positions,
        )
        next_position, hit_obstacle, hit_wall = apply_action(
            agent_position=agent_position,
            action_index=action_index,
            grid_size=environment.size,
            obstacle_positions=obstacle_positions,
        )
        agent_position = next_position
        visited_positions.add(agent_position)

        print("")
        print(f"Step {step}: obstacle={hit_obstacle} wall={hit_wall}")
        print(render_grid(build_environment_frame(environment, agent_position)))

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if agent_position == environment.target_position:
            print(f"Target reached in {step} steps.")
            return True

    print("Target was not reached within the demo step limit.")
    return False


def visualize_multiple_environments(
    policy: DirectionPolicy,
    config: TrainingConfig,
) -> list[bool]:
    """Show greedy rollouts for several post-training environments."""

    results: list[bool] = []

    for environment_index in range(config.demo_environment_count):
        demo_generator = RandomEnvironmentGenerator(
            size=config.grid_size,
            obstacle_count=config.obstacle_count,
            seed=config.seed + 1 + environment_index,
        )
        print("")
        print(
            f"=== Demo Environment {environment_index + 1}/"
            f"{config.demo_environment_count} ==="
        )
        reached_target = visualize_policy(
            policy=policy,
            environment_generator=demo_generator,
            max_steps=config.max_steps,
            sleep_seconds=config.demo_sleep_seconds,
        )
        results.append(reached_target)

    successes = sum(results)
    print("")
    print(
        f"Demo summary: reached target in {successes}/"
        f"{config.demo_environment_count} environments."
    )
    return results
