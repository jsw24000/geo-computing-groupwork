from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class SDFEncoderDecoder(nn.Module, ABC):
    """Common interface for SDF <-> latent encoder/decoder models."""

    encoder_decoder_name: str = "base"

    @abstractmethod
    def encode(self, sdf: torch.Tensor) -> torch.Tensor:
        """Encode SDF grids shaped [B, 1, D, H, W] into latent tensors."""

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors shaped [B, C, d, h, w] back to SDF grids."""

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
