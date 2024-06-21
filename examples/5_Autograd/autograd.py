#!/usr/bin/env python3
"""
Autograd demo taken from
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
"""
import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

Q = 3 * a**3 - b**2

external_grad = torch.tensor([1.0, 1.0])
Q.backward(gradient=external_grad)

assert torch.allclose(9 * a**2, a.grad)
assert torch.allclose(-2 * b, b.grad)
