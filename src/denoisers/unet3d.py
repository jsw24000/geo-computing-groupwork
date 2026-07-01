from __future__ import annotations

import math
from typing import Any

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
    """Residual 3D conv block with time embedding bias injection."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        time_bias = self.time_proj(time_embedding).view(time_embedding.shape[0], -1, 1, 1, 1)
        h = h + time_bias
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class AttnBlock3D(nn.Module):
    """3D self-attention at bottleneck resolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.q = nn.Conv3d(channels, channels, kernel_size=1)
        self.k = nn.Conv3d(channels, channels, kernel_size=1)
        self.v = nn.Conv3d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, d, h_dim, w = q.shape
        q = q.reshape(b, c, d * h_dim * w).permute(0, 2, 1)  # b, n, c
        k = k.reshape(b, c, d * h_dim * w)                    # b, c, n
        attn = torch.bmm(q, k) * (c ** -0.5)                  # b, n, n
        attn = torch.softmax(attn, dim=-1)

        v = v.reshape(b, c, d * h_dim * w)                    # b, c, n
        h_ = torch.bmm(v, attn.permute(0, 2, 1))              # b, c, n
        h_ = h_.reshape(b, c, d, h_dim, w)
        h_ = self.proj_out(h_)
        return residual + h_


class UNet3D(nn.Module):
    """Enhanced 3D U-Net denoiser for latent DDPM.

    Architecture:
        in_block → down1 (stride-2, 16→8)
                 → down2 (stride-2, 8→4)
                 → mid_block x2 + AttnBlock
                 → up2 (stride-2, 4→8, concat skip2)
                 → up1 (stride-2, 8→16, concat skip1)
                 → out_block → out_conv
    """

    def __init__(self, channels: int, base_channels: int = 128, time_dim: int = 512):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        c = base_channels  # 128

        # Input
        self.in_block = TimeBlock3D(channels, c, time_dim)  # 3→128, 16×16×16

        # Level 1: 16×16×16 → 8×8×8
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=4, stride=2, padding=1)  # 128→256
        self.block1 = TimeBlock3D(c * 2, c * 2, time_dim)  # 256→256

        # Level 2: 8×8×8 → 4×4×4
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)  # 256→512
        self.block2 = TimeBlock3D(c * 4, c * 4, time_dim)  # 512→512

        # Bottleneck: 4×4×4 with attention
        self.mid1 = TimeBlock3D(c * 4, c * 4, time_dim)
        self.attn = AttnBlock3D(c * 4)
        self.mid2 = TimeBlock3D(c * 4, c * 4, time_dim)

        # Level 2 up: 4×4×4 → 8×8×8
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)  # 512→256
        self.up_block2 = TimeBlock3D(c * 4, c * 2, time_dim)  # concat(skip2=256, up2=256)→256

        # Level 1 up: 8×8×8 → 16×16×16
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=4, stride=2, padding=1)  # 256→128
        self.up_block1 = TimeBlock3D(c * 2, c, time_dim)  # concat(skip1=128, up1=128)→128

        # Output
        self.out_block = TimeBlock3D(c, c, time_dim)
        self.out = nn.Conv3d(c, channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del condition
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x.shape[0])
        emb = self.time_mlp(sinusoidal_embedding(timesteps, self.time_dim))

        # Encoder
        skip1 = self.in_block(x, emb)               # [B, 128, 16, 16, 16]
        h = self.down1(skip1)                       # [B, 256, 8, 8, 8]
        h = self.block1(h, emb)                      # [B, 256, 8, 8, 8]
        skip2 = h

        h = self.down2(h)                            # [B, 512, 4, 4, 4]
        h = self.block2(h, emb)                      # [B, 512, 4, 4, 4]

        # Bottleneck
        h = self.mid1(h, emb)
        h = self.attn(h)
        h = self.mid2(h, emb)

        # Decoder level 2
        h = self.up2(h)                              # [B, 256, 8, 8, 8]
        if h.shape[-3:] != skip2.shape[-3:]:
            h = torch.nn.functional.interpolate(h, size=skip2.shape[-3:], mode="trilinear", align_corners=False)
        h = torch.cat([h, skip2], dim=1)             # [B, 512, 8, 8, 8]
        h = self.up_block2(h, emb)                    # [B, 256, 8, 8, 8]

        # Decoder level 1
        h = self.up1(h)                              # [B, 128, 16, 16, 16]
        if h.shape[-3:] != skip1.shape[-3:]:
            h = torch.nn.functional.interpolate(h, size=skip1.shape[-3:], mode="trilinear", align_corners=False)
        h = torch.cat([h, skip1], dim=1)             # [B, 256, 16, 16, 16]
        h = self.up_block1(h, emb)                    # [B, 128, 16, 16, 16]

        # Output
        h = self.out_block(h, emb)
        return self.out(h)                            # [B, 3, 16, 16, 16]
