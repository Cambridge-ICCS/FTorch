"""
Train and test a Fourier Neural Operator (FNO) on a sine wave dataset.

It includes data generation, model training, evaluation, and visualization.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from fno1d import FNO1d
from torch import nn, optim


def generate_sine_data(
    batch_size: int = 16, size_x: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sine wave data.

    Creates data of size <batch_size> for training and testing the FNO1d model.
    The data consists of a grid of points and their corresponding sine values.
    Copied for stability purposes. The task is trivial, but aim is not to solve a
    challenging FNO problem, but to test the FTorch pipeline.


    Parameters
    ----------
    batch_size : int
        Number of samples in the batch.
    size_x : int
        Number of spatial points in the input.

    Returns
    -------
    input_tensor : torch.Tensor
        Input tensor of shape (batch, x, 1).
    grid_tensor : torch.Tensor
        Grid tensor of shape (batch, x, 1).
    target_tensor : torch.Tensor
        Target tensor of shape (batch, x, 1).
    """
    x = np.linspace(0, 1, size_x)
    gridx = np.expand_dims(x, axis=(0, 2))  # shape (1, size_x, 1)
    gridx = np.repeat(gridx, batch_size, axis=0)  # (batch, size_x, 1)

    dummy_u = np.zeros_like(gridx)  # dummy values

    input_tensor = torch.tensor(dummy_u, dtype=torch.float32)  # shape (batch, x, 1)
    grid_tensor = torch.tensor(gridx, dtype=torch.float32)
    target_tensor = torch.tensor(np.sin(2 * np.pi * gridx), dtype=torch.float32)

    return input_tensor, grid_tensor, target_tensor


def generate_parametric_sine_data(
    batch_size: int = 32,
    size_x: int = 32,
    random_x: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of sine waves with varying amplitude, frequency, and phase.

    The data consists of a grid of points and their corresponding sine values.
    The aim is train the network on a variety of sine waves with different parameters.
    The x-values can be either random or evenly spaced. Evenly spaced x-values are
    recommended.

    Parameters
    ----------
    batch_size : int
        Number of sine functions to generate.
    size_x : int
        Number of spatial points per sample.
    random_x : bool
        If True, use random (sorted) x-points per sample. Otherwise use linspace.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dummy : torch.Tensor
        Dummy input of shape (batch_size, size_x, 1)
    grid : torch.Tensor
        x-values of shape (batch_size, size_x, 1)
    target : torch.Tensor
        Target u(x) = A * sin(2π * f * x + φ), shape (batch_size, size_x, 1)
    """
    rng = np.random.default_rng(seed)

    dummy_batch = []
    grid_batch = []
    target_batch = []

    for _ in range(batch_size):
        # Generate grid
        if random_x:
            x = np.sort(rng.uniform(0, 1, size_x))
        else:
            x = np.linspace(0, 1, size_x)

        # Generate sine parameters
        A = rng.uniform(0.5, 1.5)
        f = rng.uniform(1.0, 3.0)
        phi = rng.uniform(0, 2 * np.pi)

        y = A * np.sin(2 * np.pi * f * x + phi)

        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, size_x, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(1, size_x, 1)
        dummy_tensor = torch.zeros_like(x_tensor)

        grid_batch.append(x_tensor)
        target_batch.append(y_tensor)
        dummy_batch.append(dummy_tensor)

    dummy = torch.cat(dummy_batch, dim=0)  # (batch_size, size_x, 1)
    grid = torch.cat(grid_batch, dim=0)
    target = torch.cat(target_batch, dim=0)

    return dummy, grid, target


def train(
    model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module
) -> nn.Module:
    """
    Train the FNO1d model on sine wave data.

    This function generates the data, initializes the model, and performs
    training using the Adam optimizer and MSE loss function.
    The model is trained for 100 epochs, and the loss is printed every 10 epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The FNO1d model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : torch.nn.Module
        The loss function to use for training.
        In this case, it is the Mean Squared Error (MSE) loss.

    Returns
    -------
    model : torch.nn.Module
       The trained FNO1d model.
    """
    model.train()
    for epoch in range(100):
        x, grid, y = generate_sine_data(batch_size=32, size_x=32)

        input_batch = torch.cat((x, grid), dim=-1)
        pred = model(input_batch)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


def validate() -> None:
    """
    Validate the FNO1d model by loading it and performing a forward pass.

    This function loads the trained model, generates fresh test data,
    performs a forward pass, and evaluates the accuracy of the prediction.
    It also plots the true and predicted sine waves.

    """
    # Load trained model
    loaded_model = FNO1d()
    loaded_model = torch.jit.load("fno1d_sine.pt")
    # model.load_state_dict(torch.load("fno1d_sine.pt"))
    loaded_model.eval()

    # Generate fresh test data
    test_input, grid, test_target = generate_sine_data(batch_size=1, size_x=32)

    # Forward pass
    with torch.no_grad():
        input_batch = torch.cat((test_input, grid), dim=-1)
        test_pred = loaded_model(input_batch)

        # Evaluate accuracy
        mse_loss = nn.MSELoss()
        test_loss = mse_loss(test_pred, test_target)

        print(f"Test MSE Loss: {test_loss.item():.6f}")
        # Plot to visualize
        x = np.linspace(0, 1, 32)
        plt.plot(
            grid.squeeze().numpy(), test_target.squeeze().numpy(), label="True sin(2πx)"
        )
        plt.scatter(
            grid.squeeze().numpy(), test_pred.squeeze().numpy(), label="Predicted"
        )
        plt.legend()
        plt.title("Sine Wave Prediction")
        plt.savefig("train_fno1d.png")

        plt.show()
        # Check if prediction is accurate
        threshold = 1e-3
        if test_loss.item() >= threshold:
            raise ValueError("Test failed: Prediction not accurate.")  # noqa: EM101
        else:
            print("Test passed: Predicted sine wave accurately!")


def main() -> None:
    """
    Train and evaluate the FNO1d model.

    Training and evaluation of the FNO1d model on a sine wave dataset.
    The model is trained for 100 epochs, and the loss is printed every 10 epochs.
    """
    model = FNO1d()
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model = train(model, optimizer, loss_fn)
    model.eval()
    validate()

    # Either save the model as a TorchScript or state_dict.

    # We can go from TorchScript directly to fno1d_infer_python.py
    # Save trained model (TorchScript)
    model.eval()
    scripted_model = torch.jit.script(model)
    model_path = "saved_fno1d_model_cpu.pt"
    scripted_model.save(model_path)

    # We can go from TorchScript to state_dict for compatibility with pt2ts.py
    # and then to fno1d_infer_python.py

    # uncomment the following if using pt2ts.py
    # Save trained model (state_dict)
    # model_path = "fno1d_sine_state_dict.pt"
    # torch.save(model.state_dict(), model_path)
    # print(f"Saved trained model to {model_path}")

    # Evaluate the model
    validate()


if __name__ == "__main__":
    main()
