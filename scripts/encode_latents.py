from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from common import project_path
from src.sdf_encoder_decoder import SDFusionVQVAEEncoderDecoder
from src.pipelines.latent_cache import LatentRecord, save_latent_record
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode preprocessed SDF grids with the frozen VQ-VAE and save latent cache."
    )
    parser.add_argument("--config", default="configs/vqvae_sdfusion.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None, help="Override VQ-VAE checkpoint path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    return parser.parse_args()


def compute_latent_stats(latent_dir: Path) -> dict:
    """Compute global mean and std over all latent files in a directory."""
    files = sorted(latent_dir.glob("*.pt"))
    if not files:
        print(f"  [!] No latent files found in {latent_dir}, skipping stats.")
        return {}

    print(f"[*] Computing global stats over {len(files)} latents ...")
    n = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for f in tqdm(files, desc="Computing stats"):
        data = torch.load(f, map_location="cpu")
        latent = data["latent"] if isinstance(data, dict) else data
        n += latent.numel()
        sum_x += latent.sum().item()
        sum_x2 += (latent ** 2).sum().item()

    mean = sum_x / n
    std = math.sqrt(max(sum_x2 / n - mean ** 2, 1e-10))
    stats = {"mean": float(mean), "std": float(std), "n_samples": len(files)}
    stats_path = latent_dir.parent / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[*] Latent stats: mean={mean:.6f}, std={std:.6f}  →  saved to {stats_path}")
    return stats


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    paths = config["paths"]

    # Resolve paths
    sdf_root = project_path(paths["sdf_root"])
    latent_root = project_path(paths["latent_root"])
    metadata_root = project_path(paths["metadata_root"])
    ensure_dirs(latent_root)

    checkpoint_path = project_path(args.checkpoint or paths["vqvae_checkpoint"])
    encoder_decoder_name = config["sdf_encoder_decoder"]["name"]

    # Load VQ-VAE adapter
    print(f"[*] Loading VQ-VAE from checkpoint: {checkpoint_path}")
    encoder_decoder = SDFusionVQVAEEncoderDecoder(
        checkpoint_path=checkpoint_path,
        config_path=project_path("configs/vqvae_snet.yaml"),
        freeze=True,
    ).to(device)
    # Trigger model loading
    _ = encoder_decoder.encode(torch.zeros(1, 1, 64, 64, 64, device=device))
    print("[+] VQ-VAE loaded and frozen.")

    # Read SDF metadata for this split
    manifest_path = metadata_root / f"sdf_chair_{args.split}.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"SDF manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if args.limit is not None:
        entries = entries[: args.limit]

    print(f"[*] Processing {len(entries)} samples from {args.split} split ...")

    latent_split_dir = latent_root / args.split
    latent_split_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    for entry in tqdm(entries, desc=f"Encoding {args.split}"):
        model_id = entry["model_id"]
        sdf_rel_path = entry["sdf_path"].replace("\\", "/")  # normalize Windows paths for Linux
        sdf_path = project_path(sdf_rel_path)

        # Support both relative-to-project and absolute paths in metadata
        if not sdf_path.exists():
            sdf_path = Path(sdf_rel_path)
        if not sdf_path.exists():
            tqdm.write(f"  [!] SDF not found: {sdf_rel_path}, skipping {model_id}")
            skipped += 1
            continue

        # Load SDF tensor
        try:
            loaded = torch.load(sdf_path, map_location="cpu")
            if isinstance(loaded, dict) and "sdf" in loaded:
                sdf_tensor = loaded["sdf"].float()
            elif isinstance(loaded, torch.Tensor):
                sdf_tensor = loaded.float()
            else:
                tqdm.write(f"  [!] Unexpected data format in {sdf_path}, skipping")
                skipped += 1
                continue
        except Exception as e:
            tqdm.write(f"  [!] Failed to load {sdf_path}: {e}, skipping")
            skipped += 1
            continue

        # Align to [B, C, D, H, W]
        if sdf_tensor.ndim == 3:
            sdf_tensor = sdf_tensor.unsqueeze(0).unsqueeze(0)
        elif sdf_tensor.ndim == 4:
            sdf_tensor = sdf_tensor.unsqueeze(0)
        sdf_tensor = sdf_tensor.to(device)

        # Encode
        with torch.no_grad():
            latent = encoder_decoder.encode(sdf_tensor)  # [1, 3, 16, 16, 16]
        latent_compact = latent.squeeze(0).cpu()  # [3, 16, 16, 16]

        # Save as LatentRecord
        record = LatentRecord(
            latent=latent_compact,
            category=entry.get("category", "chair"),
            model_id=model_id,
            sdf_path=str(sdf_path),
            encoder_decoder_name=encoder_decoder_name,
            encoder_decoder_checkpoint=str(checkpoint_path),
        )
        out_path = latent_split_dir / f"{model_id}.pt"
        save_latent_record(record, out_path)
        success += 1

    print(f"\n[###] Done: {success} encoded, {skipped} skipped.")
    print(f"[*] Latents saved to: {latent_split_dir}")

    # Compute global stats for the train split (used for DDPM normalization)
    if args.split == "train" and success > 0:
        compute_latent_stats(latent_split_dir)


if __name__ == "__main__":
    main()
