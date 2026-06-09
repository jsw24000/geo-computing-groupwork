from __future__ import annotations

import argparse

import torch

from common import project_path
from src.diffusion.gaussian_diffusion import GaussianDiffusion3D
from src.models.unet3d import UNet3D
from src.models.vae import SDFVAE
from src.utils.config import ensure_dirs, load_config
from src.utils.mesh import sdf_to_mesh
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a mesh from a trained or smoke-test latent diffusion checkpoint.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/smoke_vae_unet.pt")
    parser.add_argument("--output", default="outputs/meshes/sample_generated.ply")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    ensure_dirs(config["outputs"]["meshes"])

    vae_cfg = config["vae"]
    diff_cfg = config["diffusion"]
    checkpoint = torch.load(project_path(args.checkpoint), map_location=device)
    latent_shape = tuple(checkpoint.get("latent_shape", (int(vae_cfg["latent_channels"]), 4, 4, 4)))

    vae = SDFVAE(
        in_channels=int(vae_cfg["in_channels"]),
        base_channels=int(vae_cfg["base_channels"]),
        latent_channels=int(vae_cfg["latent_channels"]),
    ).to(device)
    denoiser = UNet3D(channels=int(latent_shape[0]), base_channels=int(diff_cfg["unet_base_channels"])).to(device)
    vae.load_state_dict(checkpoint["vae"], strict=False)
    denoiser.load_state_dict(checkpoint["denoiser"], strict=False)
    vae.eval()
    denoiser.eval()

    diffusion = GaussianDiffusion3D(
        denoiser,
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
    ).to(device)

    with torch.no_grad():
        latent = diffusion.sample((1, *latent_shape), device=device, steps=int(diff_cfg["sample_steps"]))
        sdf = vae.decode(latent).detach().cpu().numpy()[0, 0]

    mesh_path = sdf_to_mesh(sdf, project_path(args.output))
    print(f"generated mesh: {mesh_path}")


if __name__ == "__main__":
    main()

