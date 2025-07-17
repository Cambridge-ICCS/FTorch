"""Replicates tests of test_optim.pf to generate numerical results for comparison."""

import torch


def one_step(optim_key):
    """Run a single optimization step with learning rate 1.0."""
    target = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
    )
    data = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)

    if optim_key == "SGD":
        optim = torch.optim.SGD([data], lr=1.0)
    elif optim_key == "Adam":
        optim = torch.optim.Adam([data], lr=1.0)
    elif optim_key == "AdamW":
        optim = torch.optim.AdamW([data], lr=1.0)

    loss = ((data - target) ** 2).mean()
    loss.backward()
    optim.step()

    return data


def four_step(optim_key):
    """Run a 4-step optimization loop varying parameters from default."""
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    data = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)

    if optim_key == "SGD":
        optim = torch.optim.SGD([data], lr=0.1, momentum=0.9)
    elif optim_key == "Adam":
        optim = torch.optim.Adam([data], lr=0.1, betas=(0.75, 0.8))
    elif optim_key == "AdamW":
        optim = torch.optim.AdamW([data], lr=0.1, betas=(0.75, 0.8))

    for _ in range(4):
        loss = ((data - target) ** 2).mean()
        loss.backward()
        optim.step()

    return data


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    # Run with variable parameters:
    print("\nResults for the `test_torch_optim_step()` unit test")

    sgd_1_res = one_step("SGD")
    print(f"   sgd_one result: {sgd_1_res}")

    adam_1_res = one_step("Adam")
    print(f"  adam_one result: {adam_1_res}")

    adamw_1_res = one_step("AdamW")
    print(f" AdamW_one result: {adamw_1_res}")

    # Run with variable parameters:
    print("\nResults for the `test_torch_optim_params()` unit test")

    sgd_4_res = four_step("SGD")
    print(f"  sgd_four result: {sgd_4_res}")

    adam_4_res = four_step("Adam")
    print(f" adam_four result: {adam_4_res}")

    adamw_4_res = four_step("AdamW")
    print(f"AdamW_four result: {adamw_4_res}")
