from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from common import project_path
from src.data.latent_dataset import LatentCacheDataModule, load_latent_stats
from src.denoisers import UNet3D
from src.diffusion.gaussian_diffusion import GaussianDiffusion3D
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent DDPM on cached VQ-VAE latents.")
    parser.add_argument("--config", default="configs/train_ddpm.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    return parser.parse_args()


@torch.no_grad()
def evaluate(val_loader: torch.utils.data.DataLoader, diffusion: GaussianDiffusion3D, device: torch.device) -> float:
    """Compute average validation loss over the val set."""
    diffusion.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        latents = batch.to(device)  # [B, C, D, H, W]
        t = torch.randint(0, diffusion.timesteps, (latents.shape[0],), device=device)
        loss_dict = diffusion.p_losses(latents, t)
        total_loss += loss_dict["loss"].item()
        num_batches += 1
    diffusion.train()
    return total_loss / max(num_batches, 1)


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    paths = config["paths"]

    latent_root = project_path(paths["latent_root"])
    checkpoint_dir = project_path(paths["checkpoint_dir"])
    log_dir = project_path(paths["log_dir"])
    ensure_dirs(checkpoint_dir, log_dir)

    # Log file
    log_file = log_dir / "train_log.jsonl"
    log_handle = log_file.open("a", encoding="utf-8") if not log_file.exists() else log_file.open("a", encoding="utf-8")

    # --- Build denoiser and diffusion ---
    denoiser_cfg = config["denoiser"]
    diff_cfg = config["diffusion"]
    train_cfg = config["training"]

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
        schedule=diff_cfg.get("beta_schedule", "linear"),
    ).to(device)

    # --- Data (with latent normalization) ---
    data_cfg = config["data"]
    stats = load_latent_stats(latent_root / "stats.json")
    if stats is not None:
        print(f"[*] Latent normalization: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    else:
        print("[*] No latent stats found — training without normalization.")
    dm = LatentCacheDataModule(
        latent_root=latent_root,
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        stats=stats,
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"[*] Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # --- Optimizer ---
    optimizer = optim.AdamW(denoiser.parameters(), lr=float(train_cfg["learning_rate"]))

    # --- Resume ---
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(project_path(args.resume), map_location=device)
        denoiser.load_state_dict(checkpoint["denoiser"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"[*] Resumed from step {start_step}")

    # --- Training loop ---
    max_steps = int(train_cfg["max_steps"])
    save_every = int(train_cfg["save_every"])
    print(f"[*] Starting training for {max_steps} steps ...")
    diffusion.train()
    start_time = time.time()

    # Infinite iterator over the train loader
    train_iter = iter(train_loader)
    pbar = tqdm(total=max_steps - start_step, desc="Training", unit="step", initial=start_step)
    step = start_step

    while step < max_steps:
        # Refresh train iterator when exhausted
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        latents = batch.to(device)  # [B, 3, 16, 16, 16]
        t = torch.randint(0, diffusion.timesteps, (latents.shape[0],), device=device)

        loss_dict = diffusion.p_losses(latents, t)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.6f}")

        # Periodic validation and checkpoint
        if step % save_every == 0 or step == max_steps:
            val_loss = evaluate(val_loader, diffusion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            elapsed = time.time() - start_time
            log_entry = {
                "step": step,
                "train_loss": round(loss.item(), 6),
                "val_loss": round(val_loss, 6),
                "best_val_loss": round(best_val_loss, 6),
                "elapsed_sec": int(elapsed),
            }
            log_handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            log_handle.flush()
            pbar.write(
                f"  Step {step:>6d} | train_loss: {loss.item():.6f} | "
                f"val_loss: {val_loss:.6f} | best: {best_val_loss:.6f}"
            )

            # Save checkpoint (only latest.pt + best.pt to save disk space)
            ckpt_payload = {
                "step": step,
                "denoiser": denoiser.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "config": config,
                "latent_stats": stats,
            }
            latest_path = checkpoint_dir / "latest.pt"
            torch.save(ckpt_payload, latest_path)

            if val_loss == best_val_loss:
                best_path = checkpoint_dir / "best.pt"
                torch.save(ckpt_payload, best_path)
                pbar.write(f"  [*] Checkpoint saved: {latest_path} (best so far)")
            else:
                pbar.write(f"  [*] Checkpoint saved: {latest_path}")

    pbar.close()
    elapsed = time.time() - start_time
    print(f"\n[###] Training complete in {timedelta(seconds=int(elapsed))}")
    print(f"     Best val loss: {best_val_loss:.6f}")
    log_handle.close()


if __name__ == "__main__":
    main()
