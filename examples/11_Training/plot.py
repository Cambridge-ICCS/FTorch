"""Plot Fortran training loss."""

import sys

import matplotlib.pyplot as plt
import numpy as np

with open("losses_ftorch.dat", "r") as f:
    ftorch = np.array([float(line) for line in f.readlines()])

# Create the comparison plot
fig, axes = plt.subplots(figsize=(10, 6))
axes.loglog(np.arange(1, len(ftorch) + 1), ftorch, ":x", color="purple", label="FTorch")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()
plt.savefig("losses.png", bbox_inches="tight")
