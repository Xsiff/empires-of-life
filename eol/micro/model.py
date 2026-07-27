"""Neural network models for micro-level agent control."""

from __future__ import annotations

import torch
from torch import nn


class DirectionPolicy(nn.Module):
    """Predicts action preferences for moving an agent on the grid."""

    def __init__(
        self, input_dim: int = 31, hidden_dim: int = 64, action_dim: int = 5
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action logits for a batch of encoded states."""

        x = self.lin1(state)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        return x


class ActorCriticPolicy(nn.Module):
    """Predicts action logits and a state value for PPO training."""

    def __init__(
        self, input_dim: int = 31, hidden_dim: int = 64, action_dim: int = 5
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        x = self.lin1(state)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        return x

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and state values for a batch of states."""

        x = self._encode(state)
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values
