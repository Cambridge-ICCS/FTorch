"""Optimizers demo."""

import torch

# We define:
#  - the input as as a vector of ones,
#  - the target as a vector where each element is the index value,
#  - a tensor to transform from input to target by elementwise multiplication
#    initialised as a vector of ones
# This is a contrived example, but provides a simple demo of optimizer functionality
input_vec = torch.ones(4)
target_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
scaling_tensor = torch.ones(4, requires_grad=True)

# Set the optimizer as torch's stochastic gradient descent (SGD)
# The parameters to tune will be the values of `tensor`, and we also set a learning rate
# Since this is a simple elemetwise example we can get away with a large learning rate
optimizer = torch.optim.SGD([scaling_tensor], lr=1.0)

# Training loop
# Run n_iter times printing every n_print steps
n_iter = 15
n_print = 1
for epoch in range(n_iter + 1):
    # Zero any previously stored gradients ready for a new iteration
    optimizer.zero_grad()

    # Forward pass: multiply the input of ones by the tensor (elementwise)
    output = input_vec * scaling_tensor

    # Create a loss tensor as computed mean square error (MSE) between target and input
    # Then perform backward step on loss to propogate gradients using autograd
    #
    # We could use the following 2 lines to do this by explicitly specifying a
    # gradient of ones to start the process:
    # loss = ((output - target) ** 2) / 4.0
    # loss.backward(gradient=torch.ones(4))
    #
    # However, we can avoid explicitly passing an initial gradient and instead do this
    # implicitly by aggregating the loss vector into a scalar value:
    loss = ((output - target_vec) ** 2).mean()
    loss.backward()

    # Step the optimizer to update the values in `tensor`
    optimizer.step()

    if (epoch) % n_print == 0:
        print(f"========================")
        print(f"Epoch: {epoch}")
        print(f"\tOutput:\n\t\t{output}")
        print(f"\tloss:\n\t\t{loss}")
        print(f"\ttensor gradient:\n\t\t{scaling_tensor.grad}")
        print(f"\tscaling_tensor:\n\t\t{scaling_tensor}")

print("Training complete.")
