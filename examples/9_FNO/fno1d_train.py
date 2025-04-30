"""
Train and test a Fourier Neural Operator (FNO) on a sine wave dataset.

It includes data generation, model training, evaluation, and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from fno1d import FNO1d
from torch import nn, optim


def generate_sine_data(batch_size=16, size_x=32):
    """
    Generate sine wave data for training.

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

    dummy_u = np.zeros_like(gridx)  # dummy values, not used really

    input_tensor = torch.tensor(dummy_u, dtype=torch.float32)  # shape (batch, x, 1)
    grid_tensor = torch.tensor(gridx, dtype=torch.float32)
    target_tensor = torch.tensor(
        np.sin(2 * np.pi * gridx), dtype=torch.float32
    )  # shape (batch, x, 1)

    return input_tensor, grid_tensor, target_tensor


def train(model, optimizer, loss_fn):
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


def validate():
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

        # Check if prediction is accurate
        threshold = 1e-3
        if test_loss.item() >= threshold:
            raise ValueError("Test failed: Prediction not accurate.") # noqa: EM101
        else:
            print("Test passed: Predicted sine wave accurately!")

        # Plot to visualize
        x = np.linspace(0, 1, 32)
        plt.plot(x, test_target.squeeze().numpy(), label="True sin(2πx)")
        plt.plot(x, test_pred.squeeze().numpy(), "--", label="Predicted")
        plt.legend()
        plt.title("Sine Wave Prediction")
        plt.savefig("train_fno1d.png")

        plt.show()

def main():
    """
    Train and evaluate the FNO1d model.

    Training and evaluation of the FNO1d model on a sine wave dataset.
    The model is trained for 100 epochs, and the loss is printed every 10 epochs.
    """
    model = FNO1d()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    model = train(model, optimizer, loss_fn)

    # Save trained model (TorchScript)
    model.eval()
    scripted_model = torch.jit.script(model)
    model_path = "fno1d_sine.pt"
    scripted_model.save(model_path)
    print(f"Saved trained model to {model_path}")

    # Evaluate the model
    validate()


if __name__ == "__main__":
    main()
