"""Random environment generation for grid-based simulations."""

from __future__ import annotations

from enum import Enum
import random


class CellType(Enum):
    """Supported cell contents for the environment grid."""

    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2
    TARGET = 3


class Environment:
    """Generated environment state."""

    def __init__(
        self,
        size: int,
        agent_position: tuple[int, int],
        target_position: tuple[int, int],
        obstacle_positions: tuple[tuple[int, int], ...],
    ) -> None:

        self.size = size
        self.agent_position = agent_position
        self.target_position = target_position
        self.obstacle_positions = obstacle_positions
        self.grid: list[list[CellType]]

        self._is_generated: bool = False

    def generate_grid(self) -> None:
        """Generate a grid representation of the environment."""

        grid = [[CellType.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        agent_pos_x, agent_pos_y = self.agent_position
        target_pos_x, target_pos_y = self.target_position

        grid[agent_pos_x][agent_pos_y] = CellType.AGENT
        grid[target_pos_x][target_pos_y] = CellType.TARGET

        for pos_x, pos_y in self.obstacle_positions:
            grid[pos_x][pos_y] = CellType.OBSTACLE

        self.grid = grid
        self._is_generated = True


class RandomEnvironmentGenerator:
    """Generates square grid environments with randomized contents."""

    def __init__(
        self, size: int, obstacle_count: int = 0, seed: int | None = None
    ) -> None:

        self.size = size
        self.obstacle_count = obstacle_count
        self._random = random.Random(seed)

    def generate_environment(self) -> Environment:
        """Generate a new environment with random non-overlapping placements."""

        positions = [(row, col) for row in range(self.size) for col in range(self.size)]
        selected_positions = self._random.sample(positions, k=self.obstacle_count + 2)

        agent_pos_x, agent_pos_y = selected_positions[0]
        target_pos_x, target_pos_y = selected_positions[1]
        obstacle_positions = tuple(selected_positions[2:])

        return Environment(
            size=self.size,
            agent_position=(agent_pos_x, agent_pos_y),
            target_position=(target_pos_x, target_pos_y),
            obstacle_positions=obstacle_positions,
        )
