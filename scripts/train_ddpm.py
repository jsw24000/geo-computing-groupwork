from __future__ import annotations

import argparse

import torch

from common import project_path
from src.denoisers import UNet3D
from src.diffusion.gaussian_diffusion import GaussianDiffusion3D
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent DDPM on cached SDFusion VQ-VAE latents.")
    parser.add_argument("--config", default="configs/train_ddpm.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    paths = config["paths"]
    ensure_dirs(project_path(paths["checkpoint_dir"]), project_path(paths["log_dir"]))

    denoiser_cfg = config["denoiser"]
    diff_cfg = config["diffusion"]
    denoiser = UNet3D(
        channels=int(denoiser_cfg["latent_channels"]),
        base_channels=int(denoiser_cfg["base_channels"]),
        time_dim=int(denoiser_cfg["time_dim"]),
    ).to(device)
    diffusion = GaussianDiffusion3D(
        denoiser,
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
    ).to(device)
    del diffusion
    raise NotImplementedError(
        "DDPM training needs a latent-cache dataset. "
        "Implement data/latents loading before starting real training."
    )


if __name__ == "__main__":
    main()

