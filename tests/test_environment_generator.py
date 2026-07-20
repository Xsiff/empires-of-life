import pytest

from eol.environment import CellType, RandomEnvironmentGenerator


def test_generator_creates_expected_grid_shape() -> None:
    environment = RandomEnvironmentGenerator(
        size=4, obstacle_count=3, seed=7
    ).generate_environment()
    environment.generate_grid()

    assert environment.size == 4
    assert len(environment.grid) == 4
    assert all(len(row) == 4 for row in environment.grid)


def test_generator_places_one_agent_one_target_and_requested_obstacles() -> None:
    environment = RandomEnvironmentGenerator(
        size=5, obstacle_count=6, seed=11
    ).generate_environment()
    environment.generate_grid()
    flattened_grid = [cell for row in environment.grid for cell in row]

    assert flattened_grid.count(CellType.AGENT) == 1
    assert flattened_grid.count(CellType.TARGET) == 1
    assert flattened_grid.count(CellType.OBSTACLE) == 6


def test_generator_keeps_positions_unique() -> None:
    environment = RandomEnvironmentGenerator(
        size=3, obstacle_count=4, seed=3
    ).generate_environment()
    all_positions = {
        environment.agent_position,
        environment.target_position,
        *environment.obstacle_positions,
    }

    assert len(all_positions) == 6


def test_generator_rejects_invalid_configuration() -> None:
    generator = RandomEnvironmentGenerator(size=2, obstacle_count=3, seed=1)

    with pytest.raises(ValueError, match="Sample larger than population"):
        generator.generate_environment()
