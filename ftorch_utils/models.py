"""Reusable PyTorch models used by FTorch examples and tests."""

import torch
from torch import nn


class SimpleNet(nn.Module):
    """PyTorch module multiplying an input vector by 2."""

    def __init__(self) -> None:
        super().__init__()
        self._fwd_seq = nn.Sequential(
            nn.Linear(5, 5, bias=False),
        )
        with torch.no_grad():
            self._fwd_seq[0].weight = nn.Parameter(2.0 * torch.eye(5))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Pass batch through the model."""
        return self._fwd_seq(batch)


class BatchingNet(nn.Module):
    """PyTorch module multiplying each input feature by a distinct scalar."""

    def __init__(self) -> None:
        super().__init__()
        self._fwd_seq = nn.Sequential(
            nn.Linear(5, 5, bias=False),
        )
        with torch.inference_mode():
            self._fwd_seq[0].weight = nn.Parameter(
                torch.diag(torch.arange(5, dtype=torch.float32))
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Pass batch through the model."""
        return self._fwd_seq(batch)


class MultiIONet(nn.Module):
    """PyTorch module multiplying two input vectors by 2 and 3."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 4, bias=False)
        self.linear2 = nn.Linear(4, 4, bias=False)
        with torch.inference_mode():
            self.linear1.weight = nn.Parameter(2.0 * torch.eye(4))
            self.linear2.weight = nn.Parameter(3.0 * torch.eye(4))

    def forward(
        self,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass the two batches through the model."""
        return self.linear1(batch1), self.linear2(batch2)