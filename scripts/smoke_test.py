from __future__ import annotations

import argparse
import math

import torch

from common import project_path
from src.data.synthetic_sdf import make_synthetic_batch
from src.diffusion.gaussian_diffusion import GaussianDiffusion3D
from src.models.unet3d import UNet3D
from src.models.vae import SDFVAE
from src.utils.config import ensure_dirs, load_config
from src.utils.mesh import sdf_to_mesh
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal VAE + latent diffusion smoke test.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))

    outputs = config["outputs"]
    ensure_dirs(outputs["checkpoints"], outputs["meshes"], outputs["figures"])

    data_cfg = config["data"]
    vae_cfg = config["vae"]
    diff_cfg = config["diffusion"]

    sdf = make_synthetic_batch(
        batch_size=int(data_cfg["batch_size"]),
        resolution=int(data_cfg["resolution"]),
        truncation=float(data_cfg["truncation"]),
        device=device,
    )
    assert sdf.shape == (int(data_cfg["batch_size"]), 1, int(data_cfg["resolution"]), int(data_cfg["resolution"]), int(data_cfg["resolution"]))

    vae = SDFVAE(
        in_channels=int(vae_cfg["in_channels"]),
        base_channels=int(vae_cfg["base_channels"]),
        latent_channels=int(vae_cfg["latent_channels"]),
    ).to(device)

    vae.train()
    vae_out = vae(sdf)
    vae_loss = vae.loss(sdf, vae_out, kl_weight=float(vae_cfg["kl_weight"]))
    assert vae_out["recon"].shape == sdf.shape
    assert torch.isfinite(vae_loss["loss"])

    latent = vae_out["mu"].detach()
    denoiser = UNet3D(
        channels=latent.shape[1],
        base_channels=int(diff_cfg["unet_base_channels"]),
    ).to(device)
    diffusion = GaussianDiffusion3D(
        denoiser,
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
    ).to(device)

    t = torch.randint(0, diffusion.timesteps, (latent.shape[0],), device=device)
    diff_out = diffusion.p_losses(latent, t)
    assert diff_out["predicted_noise"].shape == latent.shape
    assert torch.isfinite(diff_out["loss"])

    sampled_latent = diffusion.sample(
        shape=tuple(latent.shape),
        device=device,
        steps=int(diff_cfg["sample_steps"]),
    )
    assert sampled_latent.shape == latent.shape

    vae.eval()
    with torch.no_grad():
        generated_sdf = vae.decode(sampled_latent).detach().cpu().numpy()[0, 0]
        recon_sdf = vae_out["recon"].detach().cpu().numpy()[0, 0]
        input_sdf = sdf.detach().cpu().numpy()[0, 0]

    generated_mesh = sdf_to_mesh(generated_sdf, project_path(outputs["meshes"], "smoke_generated.ply"))
    recon_mesh = sdf_to_mesh(recon_sdf, project_path(outputs["meshes"], "smoke_reconstruction.ply"))
    input_mesh = sdf_to_mesh(input_sdf, project_path(outputs["meshes"], "smoke_input.ply"))

    checkpoint_path = project_path(outputs["checkpoints"], "smoke_vae_unet.pt")
    torch.save(
        {
            "vae": vae.state_dict(),
            "denoiser": denoiser.state_dict(),
            "config": config,
            "latent_shape": tuple(latent.shape[1:]),
        },
        checkpoint_path,
    )

    print(f"device: {device}")
    print(f"sdf shape: {tuple(sdf.shape)}")
    print(f"latent shape: {tuple(latent.shape)}")
    vae_loss_value = float(vae_loss["loss"].detach())
    diffusion_loss_value = float(diff_out["loss"].detach())
    print(f"vae loss: {vae_loss_value:.6f}")
    print(f"diffusion loss: {diffusion_loss_value:.6f}")
    print(f"input mesh: {input_mesh}")
    print(f"reconstruction mesh: {recon_mesh}")
    print(f"generated mesh: {generated_mesh}")
    print(f"checkpoint: {checkpoint_path}")
    if not math.isfinite(vae_loss_value) or not math.isfinite(diffusion_loss_value):
        raise RuntimeError("Smoke test produced a non-finite loss.")
    print("smoke test passed")


if __name__ == "__main__":
    main()
