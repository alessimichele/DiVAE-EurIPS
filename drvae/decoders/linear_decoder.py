import torch
from torch import nn
from typing import List, Tuple
import numpy as np

from ..nn import GatedDense, NonLinear
from torch.nn import Linear

class GatedLinearDecoder(nn.Module):

    def __init__(self, 
                n_latent: int, 
                h_dims: List[int],
                output_shape: Tuple[int, ...]):
        """
        Gated linear decoder with ReLU activations.

        Args:
            n_latent (int): Number of latent features.
            h_dims (List[int]): Hidden dimensions.
            output_shape (Tuple[int]): Output shape, expected: (C, H, W).

        Returns:
            torch.Tensor: Reconstruction distribution parameters.
        """
        super().__init__()

        n_output = np.prod(output_shape)
        self.n_latent = n_latent
        self.h_dims = h_dims
        self.n_output = n_output

        self.layers = []
        in_features = n_latent
        for h_dim in h_dims:
            self.layers.append(GatedDense(in_features, h_dim))
            in_features = h_dim

        self.layers.append(NonLinear(in_features, n_output, activation=nn.Identity()))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)

class LinearDecoder(nn.Module):
    def __init__(self, 
                n_latent: int, 
                h_dims: List[int],
                output_shape: Tuple[int]):
        """
        Linear decoder with ReLU activations.

        Args:
            n_latent (int): Number of latent features.
            h_dims (List[int]): Hidden dimensions.
            output_shape (Tuple[int]): Output shape, expected: (C, H, W).

        Returns:
            torch.Tensor: Reconstruction distribution parameters.
        """
        super().__init__()

        n_output = np.prod(output_shape)
        self.n_latent = n_latent
        self.h_dims = h_dims
        self.n_output = n_output

        self.layers = []
        in_features = n_latent
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_features, h_dim))
            self.layers.append(nn.ReLU())
            in_features = h_dim

        self.layers.append(nn.Linear(in_features, n_output))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)