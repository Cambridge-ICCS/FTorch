"""Create a neural network model with multiple inputs/outputs & save to TorchScript."""

import torch

from ftorch_utils.models import MultiIONet

model = MultiIONet().eval()
scripted_model = torch.jit.script(model)
scripted_model.save("multiionet.pt")
