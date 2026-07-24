import torch



from ..environment import Environment, RandomEnvironmentGenerator, CellType
from .model import DirectionPolicy


def generate_features(environment: Environment) -> torch.Tensor:

    agent_x, agent_y = environment.agent_position
    target_x, target_y = environment.target_position
    dx, dy = (target_x - agent_x), (target_y - agent_y)

    agent_up: CellType = environment.grid[agent_x][agent_y + 1]
    agent_down: CellType = environment.grid[agent_x][agent_y - 1]
    agent_left: CellType = environment.grid[agent_x][agent_y + 1]
    agent_right: CellType = environment.grid[agent_x][agent_y + 1]


def encode_agent_surroundings(environment: Environment):




def one_episode(
        max_steps: int, 
        generator: RandomEnvironmentGenerator, 
        model: torch.nn.Module
    ):

    # generate_environment

    environment, _ = generator.generate_environment()
    environment.generate_grid()

    agent_x, agent_y = environment.agent_position
    target_x, target_y = environment.target_position
    dx, dy = (target_x - agent_x), (target_y - agent_y)

    agent_up: CellType = environment.grid[agent_x][agent_y + 1]
    agent_down: CellType = environment.grid[agent_x][agent_y + 1]
    agent_left: CellType = environment.grid[agent_x][agent_y + 1]
    agent_right: CellType = environment.grid[agent_x][agent_y + 1]









    # run model
    ...
    ...
    ...
    # collect reward
    # append reward 
