"""Create a simple neural network model and write it out using TorchScript."""

import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._fwd_seq = nn.Sequential(
            nn.Linear(5, 5, bias=False),
        )
        self._fwd_seq[0].weight = nn.Parameter(2.0 * torch.eye(5))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self._fwd_seq(batch)


model = SimpleNet().eval()
scripted_model = torch.jit.script(model)
scripted_model.save("test_model.pt")
