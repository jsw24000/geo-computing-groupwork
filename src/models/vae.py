from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SDFVAE(nn.Module):
    """Lightweight volumetric VAE for 32^3 T-SDF smoke tests."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8, latent_channels: int = 4):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, c, stride=1),
            ConvBlock3D(c, c * 2, stride=2),
            ConvBlock3D(c * 2, c * 4, stride=2),
            ConvBlock3D(c * 4, c * 4, stride=2),
        )
        self.to_mu = nn.Conv3d(c * 4, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv3d(c * 4, latent_channels, kernel_size=1)

        self.from_latent = nn.Conv3d(latent_channels, c * 4, kernel_size=1)
        self.decoder = nn.Sequential(
            ConvBlock3D(c * 4, c * 4),
            nn.ConvTranspose3d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ConvBlock3D(c * 2, c * 2),
            nn.ConvTranspose3d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ConvBlock3D(c, c),
            nn.ConvTranspose3d(c, c, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(c, in_channels, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-20.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def loss(self, x: torch.Tensor, output: dict[str, torch.Tensor], kl_weight: float = 1e-6) -> dict[str, torch.Tensor]:
        recon_loss = F.l1_loss(output["recon"], x)
        mu = output["mu"]
        logvar = output["logvar"]
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + kl_weight * kl
        return {"loss": total, "recon_loss": recon_loss, "kl_loss": kl}

    def load_checkpoint(self, path: str | Path, strict: bool = False) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        self.load_state_dict(state_dict, strict=strict)

