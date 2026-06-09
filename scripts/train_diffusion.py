from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from common import project_path
from src.data.synthetic_sdf import SyntheticSDFConfig, SyntheticSDFDataset
from src.diffusion.gaussian_diffusion import GaussianDiffusion3D
from src.models.unet3d import UNet3D
from src.models.vae import SDFVAE
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal latent diffusion model on VAE latents.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--vae-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    ensure_dirs(config["outputs"]["checkpoints"])

    data_cfg = config["data"]
    vae_cfg = config["vae"]
    diff_cfg = config["diffusion"]

    dataset = SyntheticSDFDataset(
        SyntheticSDFConfig(
            resolution=int(data_cfg["resolution"]),
            num_samples=int(data_cfg["num_samples"]),
            shape_types=tuple(data_cfg["shape_types"]),
            truncation=float(data_cfg["truncation"]),
        )
    )
    loader = DataLoader(dataset, batch_size=int(data_cfg["batch_size"]), shuffle=True)

    vae = SDFVAE(
        in_channels=int(vae_cfg["in_channels"]),
        base_channels=int(vae_cfg["base_channels"]),
        latent_channels=int(vae_cfg["latent_channels"]),
    ).to(device)
    if args.vae_checkpoint:
        vae.load_checkpoint(project_path(args.vae_checkpoint), strict=False)
    vae.eval()

    with torch.no_grad():
        first_sdf = next(iter(loader))["sdf"].to(device)
        latent_shape = vae.encode(first_sdf)[0].shape[1:]

    denoiser = UNet3D(channels=int(latent_shape[0]), base_channels=int(diff_cfg["unet_base_channels"])).to(device)
    diffusion = GaussianDiffusion3D(
        denoiser,
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
    ).to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=float(diff_cfg["learning_rate"]))

    last_loss = None
    for step, batch in zip(range(int(diff_cfg["train_steps"])), loader):
        sdf = batch["sdf"].to(device)
        with torch.no_grad():
            latent = vae.encode(sdf)[0]
        timesteps = torch.randint(0, diffusion.timesteps, (latent.shape[0],), device=device)
        losses = diffusion.p_losses(latent, timesteps)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        last_loss = float(losses["loss"].detach())
        print(f"step {step + 1}: loss={last_loss:.6f}")

    checkpoint_path = project_path(config["outputs"]["checkpoints"], "diffusion_latest.pt")
    torch.save(
        {
            "vae": vae.state_dict(),
            "denoiser": denoiser.state_dict(),
            "config": config,
            "latent_shape": tuple(latent_shape),
        },
        checkpoint_path,
    )
    print(f"saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
