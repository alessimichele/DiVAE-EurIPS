import torch
from torch import nn
from typing import List, Tuple

class MnistEncoder(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int],
                 n_latent: int,
                 **kwargs
                 ):
        super().__init__()

        assert len(input_shape) == 3, "input_shape must be a tuple (C, H, W)"

        self.layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                #nn.BatchNorm2d(num_features=32),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                #nn.BatchNorm2d(num_features=64),
                nn.ReLU(),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(num_features=128),
                nn.ReLU(),

                nn.Flatten(),
            )
        
        self.mu_encoder = nn.Linear(128*7*7, n_latent)
        self.log_var_encoder = nn.Linear(128*7*7, n_latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layers(x)
        return self.mu_encoder(h), self.log_var_encoder(h)
