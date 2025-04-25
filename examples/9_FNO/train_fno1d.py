import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from fno1d import FNO1d

# def generate_sine_data(batch_size=16, size_x=32):
#     x = np.linspace(0, 1, size_x)
#     gridx = np.expand_dims(x, axis=(0, 2))  # shape (1, size_x, 1)
#     gridx = np.repeat(gridx, batch_size, axis=0)  # (batch, size_x, 1)

#     dummy_u = np.zeros_like(gridx)  # dummy values, not used really

#     input_tensor = torch.tensor(dummy_u, dtype=torch.float32)  # shape (batch, x, 1)
#     target_tensor = torch.tensor(np.sin(2 * np.pi * gridx), dtype=torch.float32)  # shape (batch, x, 1)

#     return input_tensor, target_tensor



# def generate_random_sine_data(batch_size=16, size_x=32):
#     batch_inputs = []
#     batch_targets = []
#     for _ in range(batch_size):
#         x = np.sort(np.random.rand(size_x))  # Random, sorted grid
#         gridx = np.expand_dims(x, axis=(0, 2))  # shape (1, size_x, 1)

#         dummy_u = np.zeros_like(gridx)  # Still dummy zeros
        
#         input_tensor = torch.tensor(dummy_u, dtype=torch.float32)  # (1, size_x, 1)
#         target_tensor = torch.tensor(np.sin(2 * np.pi * x), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, size_x, 1)

#         batch_inputs.append(input_tensor)
#         batch_targets.append(target_tensor)

#     # Stack to create batch dimension
#     inputs = torch.cat(batch_inputs, dim=0)  # (batch, size_x, 1)
#     targets = torch.cat(batch_targets, dim=0)  # (batch, size_x, 1)

#     return inputs, targets

def generate_sine_data(batch_size=16, size_x=32):
    x = np.linspace(0, 1, size_x)
    gridx = np.expand_dims(x, axis=(0, 2))  # shape (1, size_x, 1)
    gridx = np.repeat(gridx, batch_size, axis=0)  # (batch, size_x, 1)

    dummy_u = np.zeros_like(gridx)  # dummy values, not used really

    input_tensor = torch.tensor(dummy_u, dtype=torch.float32)  # shape (batch, x, 1)
    grid_tensor = torch.tensor(gridx, dtype=torch.float32)
    target_tensor = torch.tensor(np.sin(2 * np.pi * gridx), dtype=torch.float32)  # shape (batch, x, 1)

    return input_tensor, grid_tensor, target_tensor

model = FNO1d()
model = model.float()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    model.train()
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


# Save trained model (TorchScript)
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("fno1d_sine.pt")
print("Saved trained model to fno1d_sine.pt")


# Plotting
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load model
loaded_model = torch.jit.load("fno1d_sine.pt")
loaded_model.eval()

# Generate fresh test data 
test_input, grid, test_target = generate_sine_data(batch_size=1, size_x=32)

# Forward pass 
with torch.no_grad():
    input_batch = torch.cat((test_input, grid), dim=-1)
    test_pred = loaded_model(input_batch)

    # --- Evaluate accuracy ---
    mse_loss = nn.MSELoss()
    test_loss = mse_loss(test_pred, test_target)

    print(f"Test MSE Loss: {test_loss.item():.6f}")

    # Check if prediction is "sufficiently accurate"
    threshold = 1e-3
    assert test_loss.item() < threshold, "Test failed: prediction not accurate enough."

    print("Test passed: predicted sine wave accurately!")

    # --- Plot to visualize ---
    x = np.linspace(0, 1, 32)
    plt.plot(x, test_target.squeeze().numpy(), label="True sin(2πx)")
    plt.plot(x, test_pred.squeeze().numpy(), '--', label="Predicted")
    plt.legend()
    plt.title("Sine Wave Prediction")
    plt.savefig("train_fno1d.png")

    plt.show()
