import pytest

from eol.environment import (
    Action,
    Agent2D,
    CellType,
    Environment,
    RandomEnvironmentGenerator,
)


def agent_positions(environment: Environment) -> set[tuple[int, int]]:
    return {agent.position for agent in environment.agents}


def first_agent(environment: Environment):
    return next(iter(environment.agents))


def agent_at(environment: Environment, position: tuple[int, int]):
    for agent in environment.agents:
        if agent.position == position:
            return agent
    raise AssertionError(f"No agent at position {position!r}")


def test_generator_creates_expected_rectangular_grid_shape() -> None:
    environment, agents = RandomEnvironmentGenerator(
        width=5,
        height=3,
        obstacle_count=3,
        seed=7,
    ).generate_environment()

    assert environment.width == 5
    assert environment.height == 3
    assert len(environment.grid) == 3
    assert all(len(row) == 5 for row in environment.grid)
    assert agents == environment.agents


def test_generator_places_requested_agents_target_and_obstacles() -> None:
    environment, agents = RandomEnvironmentGenerator(
        width=6,
        height=4,
        agent_count=2,
        obstacle_count=5,
        seed=11,
    ).generate_environment()
    flattened_grid = [cell for row in environment.grid for cell in row]

    assert flattened_grid.count(CellType.AGENT) == 2
    assert flattened_grid.count(CellType.TARGET) == 1
    assert flattened_grid.count(CellType.OBSTACLE) == 5
    assert len(agents) == 2
    assert agents == environment.agents


def test_generator_keeps_positions_unique() -> None:
    environment, _ = RandomEnvironmentGenerator(
        width=4,
        height=4,
        agent_count=2,
        obstacle_count=4,
        seed=3,
    ).generate_environment()
    all_positions = {
        *agent_positions(environment),
        environment.target_position,
        *environment.obstacle_positions,
    }

    assert len(all_positions) == 7


def test_generator_rejects_invalid_configuration() -> None:
    generator = RandomEnvironmentGenerator(
        width=2,
        height=2,
        agent_count=2,
        obstacle_count=2,
        seed=1,
    )

    with pytest.raises(
        ValueError, match="Requested entities exceed available grid cells."
    ):
        generator.generate_environment()


def test_environment_rejects_overlapping_layout() -> None:
    with pytest.raises(ValueError, match="cannot overlap the target"):
        Environment(
            width=4,
            height=4,
            agents={Agent2D((1, 1))},
            target_position=(1, 1),
            obstacle_positions=(),
        )


def test_move_agent_raises_for_blocked_move() -> None:
    environment = Environment(
        width=4,
        height=3,
        agents={Agent2D((1, 1)), Agent2D((1, 3))},
        target_position=(2, 2),
        obstacle_positions=((1, 2),),
    )

    with pytest.raises(ValueError, match="Agent cannot move there"):
        environment.move_agent((agent_at(environment, (1, 1)), Action.RIGHT))


def test_move_agent_succeeds_for_open_cell() -> None:
    environment = Environment(
        width=4,
        height=4,
        agents={Agent2D((1, 1))},
        target_position=(1, 2),
        obstacle_positions=(),
    )

    environment.move_agent((first_agent(environment), Action.DOWN))

    assert agent_positions(environment) == {(2, 1)}
    assert environment.grid[1][1] is CellType.EMPTY
    assert environment.grid[2][1] is CellType.AGENT


def test_observe_surroundings_returns_neighbor_cells() -> None:
    environment = Environment(
        width=4,
        height=4,
        agents={Agent2D((1, 1)), Agent2D((2, 1))},
        target_position=(1, 2),
        obstacle_positions=((1, 0),),
    )

    surroundings = environment.observe_surroundings(agent_at(environment, (1, 1)))

    assert surroundings[Action.LEFT] is CellType.OBSTACLE
    assert surroundings[Action.RIGHT] is CellType.TARGET
    assert surroundings[Action.UP] is CellType.EMPTY
    assert surroundings[Action.DOWN] is CellType.AGENT
    assert surroundings[Action.HALT] is CellType.AGENT


def test_move_agent_rejects_out_of_bounds_move() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(2, 2),
        obstacle_positions=(),
    )

    with pytest.raises(ValueError, match="Agent cannot move there"):
        environment.move_agent((first_agent(environment), Action.UP))


def test_agents_is_the_single_agent_access_pattern() -> None:
    environment = Environment(
        width=3,
        height=3,
        agents={Agent2D((0, 0))},
        target_position=(2, 2),
        obstacle_positions=(),
    )

    assert agent_positions(environment) == {(0, 0)}


def test_create_agent_adds_agent_to_existing_environment() -> None:
    environment = Environment(
        width=4,
        height=4,
        agents={Agent2D((0, 0))},
        target_position=(3, 3),
        obstacle_positions=((1, 1),),
    )

    created_agent = environment.create_agent((2, 2))

    assert created_agent in environment.agents
    assert agent_positions(environment) == {(0, 0), (2, 2)}
    assert environment.grid[2][2] is CellType.AGENT


def test_create_agent_rejects_non_empty_cell() -> None:
    environment = Environment(
        width=4,
        height=4,
        agents={Agent2D((0, 0))},
        target_position=(3, 3),
        obstacle_positions=((1, 1),),
    )

    with pytest.raises(ValueError, match="Agent position must be an empty cell."):
        environment.create_agent((1, 1))


def test_delete_agent_removes_agent_from_existing_environment() -> None:
    environment = Environment(
        width=4,
        height=4,
        agents={Agent2D((0, 0)), Agent2D((2, 2))},
        target_position=(3, 3),
        obstacle_positions=(),
    )

    removed_agent = agent_at(environment, (2, 2))
    deleted_agent = environment.delete_agent(removed_agent)

    assert deleted_agent is None
    assert agent_positions(environment) == {(0, 0)}
    assert environment.grid[2][2] is CellType.EMPTY


def test_delete_agent_rejects_removing_last_agent() -> None:
    environment = Environment(
        width=4,
        height=4,
        agents={Agent2D((0, 0))},
        target_position=(3, 3),
        obstacle_positions=(),
    )

    with pytest.raises(ValueError, match="Environment must contain at least one agent."):
        environment.delete_agent(first_agent(environment))
