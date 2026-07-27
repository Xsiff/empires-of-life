"""State encoding and action helpers for micro navigation."""

from __future__ import annotations

import torch

from eol.environment import Action, Agent2D, Environment


ACTION_SPACE: tuple[Action, ...] = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.HALT,
)


def encode_state(environment: Environment, agent: Agent2D) -> torch.Tensor:
    """Encode one agent's navigation state as a fixed-size tensor."""

    agent_row, agent_col = agent.position
    target_row, target_col = environment.target_position
    delta_row = target_row - agent_row
    delta_col = target_col - agent_col

    height_scale = max(environment.height - 1, 1)
    width_scale = max(environment.width - 1, 1)

    local_view_features: list[float] = []
    for row_offset in (-2, -1, 0, 1, 2):
        for col_offset in (-2, -1, 0, 1, 2):
            observed_position = (agent_row + row_offset, agent_col + col_offset)
            if not environment.in_bounds(observed_position):
                local_view_features.append(-1.0)
            elif observed_position == environment.target_position:
                local_view_features.append(0.25)
            elif observed_position in environment.obstacle_set:
                local_view_features.append(-1.0)
            elif observed_position in environment.agent_positions:
                local_view_features.append(1.0)
            else:
                local_view_features.append(0.0)

    return torch.tensor(
        [
            agent_row / height_scale,
            agent_col / width_scale,
            target_row / height_scale,
            target_col / width_scale,
            delta_row / height_scale,
            delta_col / width_scale,
            *local_view_features,
        ],
        dtype=torch.float32,
    )


def manhattan_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Return Manhattan distance between two positions."""

    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def get_valid_action_mask(environment: Environment, agent: Agent2D) -> torch.Tensor:
    """Return a binary mask for valid actions, including halt."""

    return torch.tensor(
        [
            1.0 if environment.can_move_to((agent, action)) else 0.0
            for action in ACTION_SPACE
        ],
        dtype=torch.float32,
    )


def mask_action_logits(
    logits: torch.Tensor, valid_action_mask: torch.Tensor
) -> torch.Tensor:
    """Mask invalid actions so selection ignores them."""

    invalid_mask = (
        valid_action_mask.unsqueeze(0) <= 0
        if valid_action_mask.ndim == 1
        else valid_action_mask <= 0
    )
    return logits.clone().masked_fill(invalid_mask, torch.finfo(logits.dtype).min)


def extract_policy_logits(
    policy_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Return action logits from a policy or actor-critic output."""

    if isinstance(policy_output, tuple):
        return policy_output[0]
    return policy_output


def resolve_action(
    environment: Environment, agent: Agent2D, action_index: int
) -> Action:
    """Resolve a suggested action to an executable action."""

    suggested_action = ACTION_SPACE[action_index]
    if environment.can_move_to((agent, suggested_action)):
        return suggested_action
    return Action.HALT
