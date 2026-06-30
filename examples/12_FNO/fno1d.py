"""Fourier Neural Operator for 1D problems using spectral convolution.

This module defines:
- SpectralConv1d: A 1D spectral convolution layer using Fourier transforms.
- FNO1d: A Fourier Neural Operator model built with multiple spectral
convolution blocks.
"""

from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, tensor

################################################################
# 1d Fourier Integral Operator
################################################################


# Initialize everything
def initialize(precision: torch.dtype) -> torch.nn.Module:
    """
    Download trained FNO-1D model and prepare for inference.

    Parameters
    ----------
    precision: torch.dtype
        Sets the working precision of the model.

    Returns
    -------
    model: torch.nn.Module
        Pretrained FNO-1D model
    """
    # Set working precision
    torch.set_default_dtype(precision)

    # Load a pre-trained PyTorch model
    print("Loading pre-trained ResNet-18 model...", end="")
    model = FNO1d()
    model = model.float()
    # load model from file:
    model.load_state_dict(torch.load("fno1d_sine_state_dict.pt"))

    # Switch-off some specific layers/parts of the model that behave
    # differently during training and inference
    model.eval()

    return model


class SpectralConv1d(nn.Module):
    """1D spectral convolution using Fourier transforms."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        1D Fourier layer: FFT -> linear transform -> inverse FFT.

        Parameters
        ----------
        in_channels  : int
            Input channels
        out_channels : int
            Output channels
        modes        : int
            Number of Fourier modes to multiply (<= floor(N / 2) + 1)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, _input, weights):
        """
        Complex multiplication of Fourier modes.

        Parameters
        ----------
        _input  : torch.Tensor
            [batch, in_channels, x]
        weights : torch.Tensor
            [in_channels, out_channels, x]

        Returns
        -------
        output  : torch.Tensor
            [batch, out_channels, x]
        """
        return torch.einsum("bix,iox->box", _input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, x)
            where x is the number of spatial points.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, x)
        """
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes] = self.compl_mul1d(
            x_ft[:, :, : self.modes], self.weights
        )

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


################################################################
# FNO1d: overall network
################################################################


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1D column input."""

    def __init__(
        self,
        modes: int = 9,
        width: int = 37,
        time_future: int = 1,
        time_history: int = 1,
    ):
        """
        1D Fourier Neural Operator model.

        Parameters
        ----------
        modes        : int
            Number of Fourier modes
        width        : int
            Feature width
        time_future  : int
            Number of future steps to predict
        time_history : int
            Number of past time steps used
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.time_future = time_future
        self.time_history = time_history

        self.fc0 = nn.Linear(self.time_history + 1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.time_future)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        u : torch.Tensor
            (batch, x, channels=time_history)

        Returns
        -------
        torch.Tensor
            (batch, x, channels=time_future)
        """
        # grid = self.get_grid(u.shape, u.device)
        # x = torch.cat((u, grid), dim=-1)  # Add grid as extra channel
        x = u
        print("Shape before fc0:", x.shape)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # (batch, width, x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)  # (batch, x, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

    # UNUSED
    def get_grid(self, shape: List[int], device: torch.device):
        """
        Generate normalized grid for input.

        Parameters
        ----------
        shape  : List[int]
            Shape of the input tensor
            (batch, x, channels)
        device  : torch.device
            Device to create the grid on

        Returns
        -------
        torch.Tensor
            grid tensor of shape (batch, x, 1)
        """
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(tensor(0), tensor(1), steps=size_x, device=device)
        gridx = gridx.view(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx
