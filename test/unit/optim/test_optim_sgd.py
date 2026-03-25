"""Replicates tests of test_optim_sgd.pf to generate values for comparison."""

import torch
from torch import optim


def get_sgd_result(learning_rate, momentum, dampening, weight_decay, nesterov, desc):
    """Get SGD result for specific parameters."""
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    data = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)

    optim = torch.optim.SGD(
        [data],
        lr=learning_rate,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

    for _ in range(4):
        loss = ((data - target) ** 2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()

    result = data.detach().numpy()
    print(
        f"SGDParams({learning_rate:.1f}D0, {momentum:.1f}D0, {dampening:.1f}D0, "
        f"{weight_decay:.3f}D0, {str(nesterov).lower()}, [{result[0]:.6f}D0, "
        f'{result[1]:.6f}D0, {result[2]:.6f}D0, {result[3]:.6f}D0], "{desc}"),'
    )


# Test cases that need values
test_cases = [
    (0.1, 0.0, 0.0, 0.0, False, "Basic SGD (no momentum)"),
    (0.1, 0.9, 0.9, 0.025, False, "Full dampening"),
    (0.01, 0.99, 0.0, 0.0, True, "Small LR, High momentum, Nesterov"),
    (0.5, 0.5, 0.25, 0.1, False, "Aggressive parameters"),
]

if __name__ == "__main__":
    print("Expected values for all test cases (matching Fortran naming):")
    for lr, mom, damp, wd, nesterov, desc in test_cases:
        get_sgd_result(lr, mom, damp, wd, nesterov, desc)
