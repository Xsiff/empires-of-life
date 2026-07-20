"""Neural network models for micro-level agent control."""

from __future__ import annotations

import torch
from torch import nn


class DirectionPolicy(nn.Module):
    """Predicts action preferences for moving an agent on the grid."""

    def __init__(
        self, input_dim: int = 6, hidden_dim: int = 64, action_dim: int = 4
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action logits for a batch of encoded states."""

        return self.network(state)
