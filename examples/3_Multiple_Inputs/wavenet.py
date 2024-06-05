"""Module defining the pytorch WaveNet architecture for coupling to MiMA."""

import torch
from torch import nn


class WaveNet(nn.Module):
    """Neural network architecture following Espinosa et al. (2022)."""

    def __init__(
        self,
        checkpoint,
        n_in: int = 42,
        n_out: int = 40,
        branch_dims=None,
    ) -> None:
        """
        Initialize a WaveNet model.

        Parameters
        ----------
        checkpoint: dict
            dictionary containing weights & statistics.
        n_in : int
            Number of input features.
        n_out : int
            Number of output features.
        branch_dims : Union[list, None]
            List of dimensions of the layers to include in each of the level-specific branches.

        """
        if branch_dims is None:
            branch_dims = [64, 32]

        super().__init__()

        shared = [nn.BatchNorm1d(n_in), nn.Linear(n_in, 256), nn.ReLU()]
        for _ in range(4):
            shared.extend([nn.Linear(256, 256)])
            shared.extend([nn.ReLU()])

        shared.extend([nn.Linear(256, branch_dims[0])])
        shared.extend([nn.ReLU()])

        # All data gets fed through shared, then extra layers defined in branches for each z-level
        branches = []
        for _ in range(n_out):
            args: list[nn.Module] = []
            for in_features, out_features in zip(branch_dims[:-1], branch_dims[1:]):
                args.extend([nn.Linear(in_features, out_features)])
                args.extend([nn.ReLU()])

            args.extend([nn.Linear(branch_dims[-1], 1)])
            branches.append(nn.Sequential(*args))

        self.shared = nn.Sequential(*shared)
        self.branches = nn.ModuleList(branches)

        self.shared.apply(_xavier_init)
        for branch in self.branches:
            branch.apply(_xavier_init)

        self.double()
        self.means = checkpoint["means"]
        self.stds = checkpoint["stds"]
        del checkpoint

    def forward(
        self, wind: torch.Tensor, lat: torch.Tensor, pressure: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the network to a `Tensor` of input features.

        Parameters
        ----------
        wind : torch.Tensor
            Tensor of of input wind flattened to (n_lat*n_lon, 40).
        lat : torch.Tensor
            Tensor of of input features flattened to (n_lat*n_lon, 1).
        pressure : torch.Tensor
            Tensor of of input features flattened to (n_lat*n_lon, 1).

        Returns
        -------
        output : torch.Tensor
            Tensor of predicted outputs.

        """
        Z, levels = self.shared(torch.cat((wind, lat, pressure), 1)), []

        for branch in self.branches:
            levels.append(branch(Z).squeeze())
        Y = torch.vstack(levels).T

        # Un-standardize
        Y *= self.stds
        Y += self.means
        return Y


def _xavier_init(layer: nn.Module) -> None:
    """
    Apply Xavier initialization to a layer if it is an `nn.Linear`.

    Parameters
    ----------
    layer : nn.Module
        Linear to potentially initialize.

    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
