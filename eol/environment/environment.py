"""Core 2D environment primitives."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from .agent import Action, Agent2D

type Position = tuple[int, int]
type Agent2DAction = tuple[Agent2D, Action]


class CellType(Enum):
    """Supported cell contents for the environment grid."""

    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2
    TARGET = 3

    def is_occupied(self):
        return self.value == 1 or self.value == 2


class Environment:
    """A 2D grid environment with agents, one target, and obstacles."""

    def __init__(
        self,
        width: int,
        height: int,
        agents: Iterable[Agent2D],
        target_position: Position,
        obstacle_positions: Iterable[Position],
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Environment width and height must be positive.")

        self.width = width
        self.height = height
        self.target_position = target_position
        self.obstacle_positions = set(obstacle_positions)
        self.agents = set(agents)
        self.agent_positions: dict[Position, Agent2D] = {}
        self._validate_layout()
        self._refresh_agent_positions()
        self.grid: list[list[CellType]] = self._generate_grid()

    @property
    def obstacle_set(self) -> set[Position]:
        """Return the obstacle positions as a set for fast membership checks."""

        return set(self.obstacle_positions)

    def _refresh_agent_positions(self) -> None:
        """Rebuild the O(1) position-to-agent lookup."""

        self.agent_positions = {agent.position: agent for agent in self.agents}

    def _validate_layout(self) -> None:
        """Validate that the initial environment layout is internally consistent."""

        if not self.agents:
            raise ValueError("Environment must contain at least one agent.")

        if not self.in_bounds(self.target_position):
            raise ValueError("Target position must be within environment bounds.")

        for position in self.obstacle_positions:
            if not self.in_bounds(position):
                raise ValueError(
                    "Obstacle positions must be within environment bounds."
                )

        for agent in self.agents:
            if not self.in_bounds(agent.position):
                raise ValueError("Agent positions must be within environment bounds.")

        agent_positions = [agent.position for agent in self.agents]
        if len(set(agent_positions)) != len(agent_positions):
            raise ValueError("Agent positions must be unique.")

        if len(self.obstacle_set) != len(self.obstacle_positions):
            raise ValueError("Obstacle positions must be unique.")

        if self.target_position in self.obstacle_set:
            raise ValueError("Target position cannot overlap an obstacle.")

        if self.target_position in set(agent.position for agent in self.agents):
            raise ValueError("Agent positions cannot overlap the target.")

        if set(agent.position for agent in self.agents) & self.obstacle_set:
            raise ValueError("Agent positions cannot overlap obstacles.")

    def _generate_grid(self) -> list[list[CellType]]:
        """Build a grid view derived from the current environment state."""

        grid = [[CellType.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        target_row, target_col = self.target_position
        grid[target_row][target_col] = CellType.TARGET

        for row, col in self.obstacle_positions:
            grid[row][col] = CellType.OBSTACLE

        for agent in self.agents:
            row, col = agent.position
            grid[row][col] = CellType.AGENT

        return grid

    def _ensure_agent_position_is_available(self, position: Position) -> None:
        """Validate that a new agent can be placed at the requested position."""

        if not self.in_bounds(position):
            raise ValueError("Agent position must be within environment bounds.")

        row, col = position
        if self.grid[row][col] is not CellType.EMPTY:
            raise ValueError("Agent position must be an empty cell.")

    def in_bounds(self, position: Position) -> bool:
        """Return whether a position is inside the environment."""

        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width

    def agents_ruleset(self):
        """Apply the current proximity-based combat rules."""

        deleted_agents: set[Agent2D] = set()
        for agent in self.agents:
            agent_surroundings = self.observe_surroundings(agent)
            enemy_count = 0
            for action, cell in agent_surroundings.items():
                if cell == CellType.AGENT:
                    nearby_position = agent.next_step_position(action)
                    nearby_agent = self.agent_positions.get(nearby_position)
                    if nearby_agent is not None and nearby_agent.team != agent.team:
                        enemy_count += 1
            if enemy_count >= 2:
                deleted_agents.add(agent)

        self.delete_agents(deleted_agents)

        deleted_agents = set()
        for agent in self.agents:
            agent_surroundings = self.observe_surroundings(agent)
            enemy_count = 0
            for action, cell in agent_surroundings.items():
                if cell == CellType.AGENT:
                    nearby_position = agent.next_step_position(action)
                    nearby_agent = self.agent_positions.get(nearby_position)
                    if nearby_agent is not None and nearby_agent.team != agent.team:
                        enemy_count += 1
            if enemy_count >= 2:
                deleted_agents.add(agent)

        self.delete_agents(deleted_agents)

    def delete_agents(self, agents: Iterable[Agent2D]) -> None:
        """Delete multiple existing agents from the current environment."""

        unique_agents = set(agents)
        if not unique_agents:
            return

        for agent in unique_agents:
            self.delete_agent(agent)

    def observe_surroundings(self, agent: Agent2D) -> dict[Action, CellType]:
        """Return the four neighboring cells around the given agent."""

        surroundings: dict[Action, CellType] = {}
        for action in Action:
            next_position = agent.next_step_position(action)
            if not self.in_bounds(next_position):
                surroundings[action] = CellType.OBSTACLE
                continue

            next_row, next_col = next_position
            surroundings[action] = self.grid[next_row][next_col]
        return surroundings

    def can_move_to(
        self,
        action: Agent2DAction,
    ) -> bool:

        agent, move = action
        if move is Action.HALT:
            return True
        x_next, y_next = agent.next_step_position(move)
        if not self.in_bounds((x_next, y_next)):
            return False
        return not self.grid[x_next][y_next].is_occupied()

    def move_agent(self, action: Agent2DAction) -> None:
        """Attempt to move one agent and return whether the move succeeded."""
        if not self.can_move_to(action):
            raise ValueError("Agent cannot move there")

        agent, move = action
        if move is Action.HALT:
            return

        x, y = agent.position
        next_x, next_y = agent.next_step_position(move)
        del self.agent_positions[(x, y)]
        agent.move_to((next_x, next_y))
        self.agent_positions[(next_x, next_y)] = agent
        self.grid[x][y] = CellType.EMPTY
        self.grid[next_x][next_y] = CellType.AGENT

    def create_agent(self, position: Position) -> Agent2D:
        """Create a new agent in the current environment."""

        self._ensure_agent_position_is_available(position)
        agent = Agent2D(position)
        self.agents.add(agent)
        self.agent_positions[position] = agent

        row, col = position
        self.grid[row][col] = CellType.AGENT
        return agent

    def delete_agent(self, agent: Agent2D) -> None:
        """Delete an existing agent from the current environment."""

        if agent not in self.agents:
            raise ValueError("Agent does not belong to this environment.")

        row, col = agent.position
        self.agents.remove(agent)
        self.agent_positions.pop((row, col), None)
        self.grid[row][col] = CellType.EMPTY
