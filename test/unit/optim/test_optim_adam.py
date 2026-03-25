"""Replicates tests of test_optim_adam.pf to generate values for comparison."""

import torch
from torch import optim


def get_adam_result(learning_rate, beta_1, beta_2, weight_decay, eps, amsgrad, desc):
    """Get Adam result for specific parameters."""
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    data = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)

    optim = torch.optim.Adam(
        [data],
        lr=learning_rate,
        betas=(beta_1, beta_2),
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
    )

    for _ in range(4):
        loss = ((data - target) ** 2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()

    result = data.detach().numpy()
    print(
        f"AdamParams({learning_rate:.1f}D0, {beta_1:.1f}D0, {beta_2:.3f}D0, "
        f"{weight_decay:.3f}D0, {eps:.1e}D0, {str(amsgrad).lower()}, "
        f"[{result[0]:.6f}D0, {result[1]:.6f}D0, {result[2]:.6f}D0, "
        f'{result[3]:.6f}D0], "{desc}"),'
    )


# Test cases that need values
test_cases = [
    (0.001, 0.9, 0.999, 0.0, 1e-8, False, "Default Adam"),
    (0.01, 0.9, 0.999, 0.01, 1e-8, False, "Higher LR, with weight decay"),
    (0.001, 0.8, 0.99, 0.0, 1e-7, False, "Different betas, different eps"),
    (0.001, 0.9, 0.999, 0.0, 1e-8, True, "With AMSGrad"),
    (0.002, 0.9, 0.999, 0.001, 1e-8, False, "Higher LR, small weight decay"),
]

if __name__ == "__main__":
    print("Expected values for all test cases (matching Fortran naming):")
    for lr, b1, b2, wd, eps, amsgrad, desc in test_cases:
        get_adam_result(lr, b1, b2, wd, eps, amsgrad, desc)
