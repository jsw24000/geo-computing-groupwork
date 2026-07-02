from __future__ import annotations

import argparse
import re
from pathlib import Path

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
    parser.add_argument("--count", type=int, default=1, help="Number of chairs to generate (auto-increment filenames)")
    parser.add_argument("--category", default=None, help="Category (e.g. chair, table, car). Scopes checkpoint and output paths.")
    return parser.parse_args()


def get_next_index(mesh_dir: Path, prefix: str) -> int:
    """Find the next index for naming: scan existing {prefix}_xxx.ply files."""
    pattern = re.compile(re.escape(prefix) + r"_(\d+)\.ply$")
    max_idx = 0
    for f in mesh_dir.glob(f"{prefix}_*.ply"):
        m = pattern.match(f.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    paths = config["paths"]

    # Resolve category
    category = args.category or config.get("data", {}).get("category", None)

    # Scope mesh and checkpoint paths by category
    mesh_dir = project_path(paths.get("mesh_dir", "outputs/meshes"))
    if category:
        mesh_dir = mesh_dir / category
    ensure_dirs(mesh_dir)

    denoiser_cfg = config["denoiser"]
    diff_cfg = config["diffusion"]
    latent_shape = tuple(int(v) for v in denoiser_cfg["latent_shape"])
    count = max(1, args.count)
    category_label = category or "chair"

    # --- Build denoiser ---
    denoiser = UNet3D(
        channels=int(denoiser_cfg["latent_channels"]),
        base_channels=int(denoiser_cfg["base_channels"]),
        time_dim=int(denoiser_cfg["time_dim"]),
    ).to(device)

    # Resolve DDPM checkpoint path (with category scoping)
    ddpm_checkpoint_path = args.ddpm_checkpoint
    if ddpm_checkpoint_path is None:
        ddpm_checkpoint_path = paths["ddpm_checkpoint"]
        if category:
            ddpm_checkpoint_path = str(Path(paths["ddpm_checkpoint"]).parent / category / Path(paths["ddpm_checkpoint"]).name)
    ddpm_checkpoint_path = project_path(ddpm_checkpoint_path)

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

    # --- Determine output numbering ---
    file_prefix = f"{category_label}_generated"
    start_idx = get_next_index(mesh_dir, prefix=file_prefix)
    sample_steps = args.sample_steps or int(diff_cfg.get("sample_steps", 100))
    method_name = "DDIM" if args.ddim else "DDPM"

    print(f"[*] Generating {count} {category_label}(s) with {method_name} ({sample_steps} steps)")
    print(f"    Files will start from {start_idx}")

    for i in range(count):
        current_idx = start_idx + i
        print(f"\n{'='*50}")
        print(f"  Sample {i+1}/{count} → {category_label}/{category_label}_generated_{current_idx:03d}.ply")
        print(f"{'='*50}")

        # --- Sample one latent ---
        with torch.no_grad():
            if args.ddim:
                latent = diffusion.ddim_sample(
                    (1, *latent_shape),
                    device=device,
                    steps=sample_steps,
                    eta=0.0,
                )
            else:
                latent = diffusion.sample(
                    (1, *latent_shape),
                    device=device,
                    steps=sample_steps,
                )
        print(f"    latent: mean={latent.mean():.4f}, std={latent.std():.4f}")

        # Reverse latent normalization
        if latent_stats is not None:
            latent = latent * latent_stats["std"] + latent_stats["mean"]

        # --- Decode to SDF ---
        with torch.no_grad():
            sdf = encoder_decoder.decode(latent)
        print(f"    SDF: shape={sdf.shape}, range=[{sdf.min():.3f}, {sdf.max():.3f}]")

        # --- Export mesh ---
        output_name = f"{category_label}_generated_{current_idx:03d}.ply"
        mesh_path = mesh_dir / output_name
        sdf_np = sdf.detach().cpu().numpy()[0, 0]
        sdf_to_mesh(sdf_np, str(mesh_path))
        print(f"    [###] Saved: {mesh_path}")

    print(f"\n[###] All {count} samples generated in: {mesh_dir}")


if __name__ == "__main__":
    main()
