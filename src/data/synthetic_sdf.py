from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


def make_grid(resolution: int, device: torch.device | str = "cpu") -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, resolution, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([xx, yy, zz], dim=0)


def sphere_sdf(grid: torch.Tensor, radius: float = 0.55, center: torch.Tensor | None = None) -> torch.Tensor:
    if center is None:
        center = torch.zeros(3, device=grid.device, dtype=grid.dtype)
    center = center.view(3, 1, 1, 1)
    return torch.linalg.norm(grid - center, dim=0) - radius


def box_sdf(grid: torch.Tensor, half_size: torch.Tensor | Iterable[float] = (0.45, 0.45, 0.45)) -> torch.Tensor:
    half_size = torch.as_tensor(half_size, device=grid.device, dtype=grid.dtype).view(3, 1, 1, 1)
    q = torch.abs(grid) - half_size
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=0)
    inside = torch.clamp(q.max(dim=0).values, max=0.0)
    return outside + inside


def truncate_sdf(sdf: torch.Tensor, truncation: float | None) -> torch.Tensor:
    if truncation is None or truncation <= 0:
        return sdf
    return torch.clamp(sdf, -truncation, truncation) / truncation


@dataclass
class SyntheticSDFConfig:
    resolution: int = 32
    num_samples: int = 8
    shape_types: tuple[str, ...] = ("sphere", "box")
    truncation: float = 0.35


class SyntheticSDFDataset(Dataset):
    """Small deterministic dataset for smoke tests and CPU-only checks."""

    def __init__(self, config: SyntheticSDFConfig):
        self.config = config
        self.grid = make_grid(config.resolution)

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        shape_type = self.config.shape_types[index % len(self.config.shape_types)]
        phase = float(index % 5) / 4.0

        if shape_type == "sphere":
            sdf = sphere_sdf(self.grid, radius=0.45 + 0.12 * phase)
        elif shape_type == "box":
            size = torch.tensor([0.35 + 0.12 * phase, 0.45, 0.4], dtype=self.grid.dtype)
            sdf = box_sdf(self.grid, half_size=size)
        else:
            raise ValueError(f"Unknown synthetic shape type: {shape_type}")

        sdf = truncate_sdf(sdf, self.config.truncation)
        return {"sdf": sdf.unsqueeze(0).float(), "label": shape_type}


def make_synthetic_batch(
    batch_size: int = 2,
    resolution: int = 32,
    truncation: float = 0.35,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    config = SyntheticSDFConfig(resolution=resolution, num_samples=batch_size, truncation=truncation)
    dataset = SyntheticSDFDataset(config)
    batch = torch.stack([dataset[i]["sdf"] for i in range(batch_size)], dim=0)
    return batch.to(device)

