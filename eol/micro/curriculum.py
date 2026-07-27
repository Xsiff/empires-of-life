"""Curriculum schedules for micro navigation training."""

from __future__ import annotations

from dataclasses import dataclass

from eol.environment import Agent2D, Environment, RandomEnvironmentGenerator
from eol.micro.config import NavigationConfig
from eol.micro.scenario import ScenarioFactory


@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum difficulty stage."""

    grid_size: int
    obstacle_count: int


CURRICULUM_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(grid_size=5, obstacle_count=1),
    CurriculumStage(grid_size=5, obstacle_count=4),
    CurriculumStage(grid_size=10, obstacle_count=4),
    CurriculumStage(grid_size=10, obstacle_count=10),
    CurriculumStage(grid_size=15, obstacle_count=10),
    CurriculumStage(grid_size=15, obstacle_count=20),
    CurriculumStage(grid_size=20, obstacle_count=20),
    CurriculumStage(grid_size=20, obstacle_count=40),
)


def get_curriculum_stage(
    episode_index: int,
    total_episodes: int,
) -> CurriculumStage:
    """Return the curriculum stage for the given training progress."""

    if total_episodes <= 1:
        return CURRICULUM_STAGES[-1]

    stage_index = min(
        (episode_index - 1) * len(CURRICULUM_STAGES) // total_episodes,
        len(CURRICULUM_STAGES) - 1,
    )
    return CURRICULUM_STAGES[stage_index]


def build_curriculum_scenario_factory(config: NavigationConfig) -> ScenarioFactory:
    """Return a staged scenario factory that grows environment difficulty."""

    def create_scenario(episode_index: int, seed: int) -> tuple[Environment, Agent2D]:
        stage = get_curriculum_stage(episode_index, config.episodes)
        generator = RandomEnvironmentGenerator(
            size=stage.grid_size,
            obstacle_count=stage.obstacle_count,
            seed=seed,
        )
        environment, agents = generator.generate_environment()
        return environment, next(iter(agents))

    return create_scenario


def get_curriculum_final_stage() -> CurriculumStage:
    """Return the final evaluation stage for the current curriculum."""

    return CURRICULUM_STAGES[-1]
