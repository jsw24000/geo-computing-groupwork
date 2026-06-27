from __future__ import annotations

import argparse

import torch

from common import project_path
from src.sdf_encoder_decoder import SDFusionVQVAEEncoderDecoder
from src.denoisers import UNet3D
from src.diffusion.gaussian_diffusion import GaussianDiffusion3D
from src.utils.config import ensure_dirs, load_config
from src.utils.mesh import sdf_to_mesh
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample latent DDPM and decode through the SDFusion VQ-VAE.")
    parser.add_argument("--config", default="configs/sample.yaml")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    paths = config["paths"]
    ensure_dirs(project_path(paths["mesh_dir"]))

    denoiser_cfg = config["denoiser"]
    diff_cfg = config["diffusion"]
    checkpoint_path = project_path(args.checkpoint or paths["ddpm_checkpoint"])
    latent_shape = tuple(int(v) for v in denoiser_cfg["latent_shape"])

    denoiser = UNet3D(
        channels=int(denoiser_cfg["latent_channels"]),
        base_channels=int(denoiser_cfg["base_channels"]),
        time_dim=int(denoiser_cfg["time_dim"]),
    ).to(device)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        denoiser.load_state_dict(checkpoint.get("denoiser", checkpoint), strict=False)
    else:
        raise FileNotFoundError(f"DDPM checkpoint not found: {checkpoint_path}")

    diffusion = GaussianDiffusion3D(
        denoiser,
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
    ).to(device)
    encoder_decoder = SDFusionVQVAEEncoderDecoder(
        checkpoint_path=project_path(paths["vqvae_checkpoint"]),
        external_root=project_path(paths["external_sdfusion"]),
        freeze=True,
    ).to(device)
    batch_size = int(config["sampling"]["batch_size"])
    with torch.no_grad():
        latent = diffusion.sample((batch_size, *latent_shape), device=device, steps=int(diff_cfg["sample_steps"]))
        sdf = encoder_decoder.decode(latent).detach().cpu().numpy()[0, 0]
    mesh_path = sdf_to_mesh(sdf, project_path(paths["mesh_dir"], config["sampling"]["output_name"]))
    print(f"generated mesh: {mesh_path}")


if __name__ == "__main__":
    main()
