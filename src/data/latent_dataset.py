from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_latent_stats(stats_path: str | Path) -> dict | None:
    """Load latent normalization stats from a JSON file."""
    stats_path = Path(stats_path)
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


class LatentCacheDataset(Dataset):
    """Dataset that loads pre-extracted VQ-VAE latent codes from disk.

    Each sample is a latent tensor of shape [C, D, H, W] (e.g. [3, 16, 16, 16])
    saved as a LatentRecord or a simple dict via torch.save/load.
    This dataset is used for unconditional DDPM training in latent space.

    If stats (dict with 'mean' and 'std') is provided, latents are normalized
    to N(0,1) via (latent - mean) / std.

    If category is provided, the latent directory becomes {latent_root}/{category}/{split}.
    Otherwise it is {latent_root}/{split} (backward compatible).
    """

    def __init__(self, latent_root: str | Path, split: str = "train", category: str | None = None, stats: dict | None = None):
        if category:
            self.latent_dir = Path(latent_root) / category / split
        else:
            self.latent_dir = Path(latent_root) / split
        if not self.latent_dir.is_dir():
            raise NotADirectoryError(f"Latent split directory not found: {self.latent_dir}")

        self.paths = sorted(self.latent_dir.glob("*.pt"))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No latent .pt files found in {self.latent_dir}")

        self.stats = stats
        if stats is not None:
            self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self.std = torch.tensor(stats["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        payload = torch.load(self.paths[index], map_location="cpu")
        if isinstance(payload, dict):
            latent = payload["latent"]
        else:
            latent = payload
        latent = latent.float()
        if self.stats is not None:
            latent = (latent - self.mean) / self.std
        return latent


class LatentCacheDataModule:
    """Convenience wrapper that creates train/val datasets from a latent root."""

    def __init__(self, latent_root: str | Path, batch_size: int = 4, num_workers: int = 0, stats: dict | None = None, category: str | None = None):
        self.latent_root = Path(latent_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stats = stats
        self.category = category

    def train_dataset(self) -> LatentCacheDataset:
        return LatentCacheDataset(self.latent_root, "train", category=self.category, stats=self.stats)

    def val_dataset(self) -> LatentCacheDataset:
        return LatentCacheDataset(self.latent_root, "val", category=self.category, stats=self.stats)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
