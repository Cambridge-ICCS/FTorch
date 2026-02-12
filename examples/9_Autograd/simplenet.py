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
scripted_model.save("simplenet.pt")

x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
y = scripted_model(x)

dydx = torch.autograd.grad(
    outputs=y,
    inputs=x,
    grad_outputs=torch.ones_like(y),
    retain_graph=True,
    allow_unused=True,
)[0]

print(f"y = {y}")
print(f"dy/dx = {dydx}")

assert torch.allclose(dydx, 2.0 * torch.ones_like(y))
