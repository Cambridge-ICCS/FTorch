"""Autograd demo taken from https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html."""

import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

Q = 3 * (a**3 - b * b / 3)
print(Q)
expect = torch.tensor([-12.0, 65.0])
if not torch.allclose(Q, expect):
    result_error = f"Result:\n{Q}\ndoes not match expected value:\n{expect}"
    raise ValueError(result_error)

external_grad = torch.tensor([1.0, 1.0])
Q.backward(gradient=external_grad)

if not torch.allclose(9 * a**2, a.grad):
    result_error = (
        f"Calculated gradient a.grad:\n{a.grad}\n"
        "does not match expected 9 * a**2\n{9 * a**2}"
    )
    raise ValueError(result_error)
if not torch.allclose(-2 * b, b.grad):
    result_error = (
        f"Calculated gradient b.grad:\n{b.grad}\n"
        "does not match expected -2 * b\n{-2 * b}"
    )
    raise ValueError(result_error)
