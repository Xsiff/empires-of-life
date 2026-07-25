"""Agent primitives for 2D grid environments."""

from __future__ import annotations

from enum import Enum


class Action(Enum):
    """Cardinal movement actions for a grid agent."""

    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP = (-1, 0)
    DOWN = (1, 0)
    HALT = (0, 0)


class Agent2D:
    """A simple 2D grid agent with a mutable position."""

    def __init__(self, position: tuple[int, int], team: int = 0) -> None:
        self.position = position
        self.team = team

    def next_step_position(self, action: Action) -> tuple[int, int]:
        """Return the next position for a proposed action."""

        delta_row, delta_col = action.value
        row, col = self.position
        return (row + delta_row, col + delta_col)

    def move_to(self, position: tuple[int, int]) -> None:
        """Move the agent to an already-validated position."""

        self.position = position
