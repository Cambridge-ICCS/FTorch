"""Create a simple neural network model and write it out using TorchScript."""

import torch

from ftorch_utils.models import SimpleNet

model = SimpleNet().eval()
scripted_model = torch.jit.script(model)
scripted_model.save("simplenet.pt")
