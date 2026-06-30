"""Plot and compare Python and Fortran optimizer losses."""

import sys

import matplotlib.pyplot as plt
import numpy as np

# Load both loss files
with open("losses_pytorch.dat", "r") as f:
    pytorch = np.array([float(line) for line in f.readlines()])

with open("losses_ftorch.dat", "r") as f:
    ftorch = np.array([float(line) for line in f.readlines()])

# Create the comparison plot
fig, axes = plt.subplots(figsize=(10, 6))
axes.loglog(
    np.arange(1, len(pytorch) + 1), pytorch, "--o", color="orange", label="PyTorch"
)
axes.loglog(np.arange(1, len(ftorch) + 1), ftorch, ":x", color="purple", label="FTorch")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()
plt.savefig("losses.png", bbox_inches="tight")

# Check if results are close (within reasonable tolerance)
try:
    np.testing.assert_allclose(pytorch, ftorch, rtol=1e-5, atol=1e-7)
    print("SUCCESS: Python and Fortran optimizer results match within tolerance")
    print(f"Python losses:  {pytorch}")
    print(f"Fortran losses: {ftorch}")
    print(f"Max difference: {np.max(np.abs(pytorch - ftorch))}")

except AssertionError as e:
    print(f"ERROR: Python and Fortran results differ: {e}")
    print(f"Python losses:  {pytorch}")
    print(f"Fortran losses: {ftorch}")
    print(f"Differences:    {pytorch - ftorch}")
