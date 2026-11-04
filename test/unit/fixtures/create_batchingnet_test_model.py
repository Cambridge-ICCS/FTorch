"""Create a batching neural network model and write it out using TorchScript."""

import torch

from ftorch_utils.models import BatchingNet

model = BatchingNet().eval()
scripted_model = torch.jit.script(model)
scripted_model.save("batchingnet.pt")
