import torch
from torch import nn
from typing import List, Tuple
import numpy as np

from ..nn import GatedDense, NonLinear
from torch.nn import Linear

class GatedLinearEncoder(nn.Module):
    
    def __init__(self, input_shape: Tuple[int],
                    h_dims: List[int],
                    n_latent: int):
            """
            Gated linear encoder with ReLU activations.
    
            Args:
                n_input (Tuple[int]): Input shape, expected: (C, H, W).
                h_dims (List[int]): Hidden dimensions.
                n_latent (int): Number of latent features.
    
            Returns:
                torch.Tensor: Latent distribution parameters.
            """        
            super().__init__()
    
            n_input = np.prod(input_shape)
    
            self.n_input = n_input
            self.h_dims = h_dims
            self.n_latent = n_latent
    
            self.layers = []
            in_features = n_input
            for h_dim in h_dims:
                self.layers.append(GatedDense(in_features, h_dim))
                in_features = h_dim
    
            self.layers = nn.Sequential(*self.layers)
    
            self.mu_encoder = Linear(self.h_dims[-1], n_latent)
            self.log_var_encoder = NonLinear(self.h_dims[-1], n_latent,  activation=nn.Hardtanh(min_val=-6.,max_val=2.))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_input)
        h = self.layers(x)
        mu = self.mu_encoder(h)
        log_var = self.log_var_encoder(h)
        return mu, log_var
    
class LinearEncoder(nn.Module):
    def __init__(self, 
                input_shape: Tuple[int], 
                h_dims: List[int],
                n_latent: int):
        """
        Linear encoder with ReLU activations.

        Args:
            n_input (Tuple[int]): Input shape, expected: (C, H, W).
            h_dims (List[int]): Hidden dimensions.
            n_latent (int): Number of latent features.

        Returns:
            torch.Tensor: Latent distribution parameters.
        """        
        super().__init__()

        n_input = np.prod(input_shape)

        self.n_input = n_input
        self.h_dims = h_dims
        self.n_latent = n_latent

        self.layers = []
        in_features = n_input
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_features, h_dim))
            self.layers.append(nn.ReLU())
            in_features = h_dim

        self.layers = nn.Sequential(*self.layers)

        self.mu_encoder = nn.Linear(self.h_dims[-1], n_latent)
        self.log_var_encoder = nn.Linear(self.h_dims[-1], n_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_input)
        h = self.layers(x)
        mu = self.mu_encoder(h)
        log_var = self.log_var_encoder(h)
        return mu, log_var