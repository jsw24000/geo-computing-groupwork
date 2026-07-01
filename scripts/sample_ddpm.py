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
    parser = argparse.ArgumentParser(description="Sample latent DDPM and decode through the VQ-VAE.")
    parser.add_argument("--config", default="configs/sample.yaml")
    parser.add_argument("--ddpm_checkpoint", default=None, help="DDPM denoiser checkpoint path")
    parser.add_argument("--vqvae_checkpoint", default=None, help="Override VQ-VAE checkpoint path")
    parser.add_argument("--ddim", action="store_true", default=False, help="Use DDIM sampling")
    parser.add_argument("--sample_steps", type=int, default=None, help="Override sampling steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    paths = config["paths"]
    mesh_dir = project_path(paths.get("mesh_dir", "outputs/meshes"))
    ensure_dirs(mesh_dir)

    denoiser_cfg = config["denoiser"]
    diff_cfg = config["diffusion"]
    latent_shape = tuple(int(v) for v in denoiser_cfg["latent_shape"])

    # --- Build denoiser ---
    denoiser = UNet3D(
        channels=int(denoiser_cfg["latent_channels"]),
        base_channels=int(denoiser_cfg["base_channels"]),
        time_dim=int(denoiser_cfg["time_dim"]),
    ).to(device)

    # Load DDPM checkpoint
    ddpm_checkpoint_path = project_path(args.ddpm_checkpoint or paths["ddpm_checkpoint"])
    if not ddpm_checkpoint_path.exists():
        raise FileNotFoundError(f"DDPM checkpoint not found: {ddpm_checkpoint_path}")
    print(f"[*] Loading DDPM checkpoint: {ddpm_checkpoint_path}")
    checkpoint = torch.load(ddpm_checkpoint_path, map_location=device)
    denoiser.load_state_dict(checkpoint.get("denoiser", checkpoint), strict=False)
    denoiser.eval()
    print(f"[+] DDPM denoiser loaded (step {checkpoint.get('step', '?')})")

    # Load latent normalization stats
    latent_stats = checkpoint.get("latent_stats", None)
    if latent_stats is not None:
        print(f"[*] Latent normalization: mean={latent_stats['mean']:.4f}, std={latent_stats['std']:.4f}")

    # --- Build diffusion ---
    diffusion = GaussianDiffusion3D(
        denoiser,
        timesteps=int(diff_cfg["timesteps"]),
        beta_start=float(diff_cfg["beta_start"]),
        beta_end=float(diff_cfg["beta_end"]),
        schedule=diff_cfg.get("beta_schedule", "linear"),
    ).to(device)

    # --- Build VQ-VAE decoder ---
    vqvae_checkpoint_path = project_path(args.vqvae_checkpoint or paths.get("vqvae_checkpoint", "saved_ckpt/vqvae-snet-all.pth"))
    encoder_decoder = SDFusionVQVAEEncoderDecoder(
        checkpoint_path=vqvae_checkpoint_path,
        config_path=project_path("configs/vqvae_snet.yaml"),
        freeze=True,
    ).to(device)

    # --- Sample ---
    batch_size = int(config["sampling"]["batch_size"])
    sample_steps = args.sample_steps or int(diff_cfg.get("sample_steps", 100))
    print(f"[*] Sampling {batch_size} latent(s) of shape {latent_shape} with {sample_steps} steps ...")
    print(f"    Method: {'DDIM' if args.ddim else 'DDPM'}")

    with torch.no_grad():
        if args.ddim:
            latent = diffusion.ddim_sample(
                (batch_size, *latent_shape),
                device=device,
                steps=sample_steps,
                eta=0.0,
            )
        else:
            latent = diffusion.sample(
                (batch_size, *latent_shape),
                device=device,
                steps=sample_steps,
            )
    print(f"[+] Sampled latent shape: {latent.shape}, mean={latent.mean():.4f}, std={latent.std():.4f}")

    # Reverse latent normalization
    if latent_stats is not None:
        latent = latent * latent_stats["std"] + latent_stats["mean"]
        print(f"[*] After reverse norm: mean={latent.mean():.4f}, std={latent.std():.4f}")

    # --- Decode to SDF ---
    print("[*] Decoding latent through VQ-VAE ...")
    with torch.no_grad():
        sdf = encoder_decoder.decode(latent)
    print(f"[+] Decoded SDF shape: {sdf.shape}, range=[{sdf.min():.3f}, {sdf.max():.3f}]")

    # --- Export mesh ---
    output_name = config["sampling"].get("output_name", "sample_generated.ply")
    mesh_path = mesh_dir / output_name
    sdf_np = sdf.detach().cpu().numpy()[0, 0]  # [D, H, W]
    sdf_to_mesh(sdf_np, str(mesh_path))
    print(f"[###] Generated mesh saved to: {mesh_path}")


if __name__ == "__main__":
    main()
