from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class TimeBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        time_bias = self.time_proj(time_embedding).view(time_embedding.shape[0], -1, 1, 1, 1)
        h = h + time_bias
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    """Small 3D U-Net denoiser for latent DDPM smoke tests."""

    def __init__(self, channels: int, base_channels: int = 16, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        c = base_channels
        self.in_block = TimeBlock3D(channels, c, time_dim)
        self.down = nn.Conv3d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.mid_block = TimeBlock3D(c * 2, c * 2, time_dim)
        self.up = nn.ConvTranspose3d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.out_block = TimeBlock3D(c * 2, c, time_dim)
        self.out = nn.Conv3d(c, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x.shape[0])
        emb = self.time_mlp(sinusoidal_embedding(timesteps, self.time_dim))
        skip = self.in_block(x, emb)
        h = self.down(skip)
        h = self.mid_block(h, emb)
        h = self.up(h)
        if h.shape[-3:] != skip.shape[-3:]:
            h = torch.nn.functional.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        h = torch.cat([h, skip], dim=1)
        h = self.out_block(h, emb)
        return self.out(h)

