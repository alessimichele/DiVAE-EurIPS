# drvae/decoders/mnist_decoder.py
import torch
from torch import nn
from typing import Tuple

class MnistDecoder(nn.Module):
    def __init__(self,
                 n_latent: int,
                 output_shape: Tuple[int, ...],
                 **kwargs):
        super().__init__()

        assert len(output_shape) == 3, "output_shape must be a tuple (C, H, W)"

        self.decoder_input = nn.Linear(n_latent, 128 * 7 * 7)

        self.layers = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 → 14x14
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 14x14 → 28x28
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Keep at 28x28
        )

    def forward(self, z: torch.Tensor):
        h = self.decoder_input(z)
        return self.layers(h)
