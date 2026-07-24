"""Random environment generation for 2D environments."""

from __future__ import annotations

import random

from .agent import Agent2D
from .environment import Environment, Position


class RandomEnvironmentGenerator:
    """Generates randomized 2D environments with non-overlapping placements."""

    def __init__(
        self,
        size: int | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        agent_count: int = 1,
        obstacle_count: int = 0,
        seed: int | None = None,
    ) -> None:
        if size is not None:
            if width is None:
                width = size
            if height is None:
                height = size

        if width is None or height is None:
            raise ValueError("Generator requires width/height or a square size.")
        if width <= 0 or height <= 0:
            raise ValueError("Generator width and height must be positive.")
        if agent_count <= 0:
            raise ValueError("Generator must create at least one agent.")
        if obstacle_count < 0:
            raise ValueError("Obstacle count cannot be negative.")

        self.width = width
        self.height = height
        self.size = width
        self.agent_count = agent_count
        self.obstacle_count = obstacle_count
        self._random = random.Random(seed)

    @property
    def total_cells(self) -> int:
        """Return the number of cells in the generator's 2D area."""

        return self.width * self.height

    def _sample_positions(self, count: int) -> list[Position]:
        """Sample unique in-bounds positions from the environment area."""

        if count > self.total_cells:
            raise ValueError("Requested entities exceed available grid cells.")

        positions = [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        ]
        return self._random.sample(positions, k=count)

    def generate_environment(self) -> tuple[Environment, set[Agent2D]]:
        """Generate a new environment and the agents placed inside it."""

        selection_size = self.agent_count + 1 + self.obstacle_count
        selected_positions = self._sample_positions(selection_size)
        agents = {
            Agent2D(position) for position in selected_positions[: self.agent_count]
        }
        target_position = selected_positions[self.agent_count]
        obstacle_positions = tuple(selected_positions[self.agent_count + 1 :])

        environment = Environment(
            width=self.width,
            height=self.height,
            agents=agents,
            target_position=target_position,
            obstacle_positions=obstacle_positions,
        )
        return environment, agents
