"""Based on https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html."""

import torch

# Construct input tensors with requires_grad=True
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

# Compute some mathematical expression
Q = 3 * (a**3 - b * b / 3)

# Reverse mode
Q.backward(gradient=torch.ones_like(Q))
print(a.grad)
print(b.grad)
