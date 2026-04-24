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
print(f"x = {x}")
y = scripted_model(x)
print(f"y = {y}")

external_gradient = torch.ones_like(y)
dydx = torch.autograd.grad(
    outputs=y,
    inputs=x,
    grad_outputs=external_gradient,
    retain_graph=True,
    allow_unused=True,
)[0]
print(f"dy/dx = {dydx}")
assert torch.allclose(dydx, 2.0 * external_gradient)

y.backward(external_gradient)

weights = model._fwd_seq[0].weight
dydw = weights.grad
print(f"dy/d(weights):\n{dydw}")
assert torch.allclose(dydw, torch.vstack(5 * [x]))
