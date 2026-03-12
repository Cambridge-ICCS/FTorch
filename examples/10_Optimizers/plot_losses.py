"""Plot losses from optimizer example."""

import matplotlib.pyplot as plt
import numpy as np

with open("losses_pytorch.dat", "r") as f:
    pytorch = [float(line) for line in f.readlines()]

with open("losses_ftorch.dat", "r") as f:
    ftorch = [float(line) for line in f.readlines()]

fig, axes = plt.subplots()
axes.loglog(np.arange(1, len(pytorch) + 1), pytorch, "--o", label="PyTorch")
axes.loglog(np.arange(1, len(ftorch) + 1), ftorch, ":x", label="FTorch")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()
plt.savefig("losses.png", bbox_inches="tight")
