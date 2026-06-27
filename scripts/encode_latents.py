from __future__ import annotations

import argparse

from common import project_path
from src.sdf_encoder_decoder import SDFusionVQVAEEncoderDecoder
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode preprocessed SDF grids with the frozen SDFusion VQ-VAE.")
    parser.add_argument("--config", default="configs/vqvae_sdfusion.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    paths = config["paths"]
    ensure_dirs(project_path(paths["latent_root"]), project_path(paths["metadata_root"]))
    encoder_decoder = SDFusionVQVAEEncoderDecoder(
        checkpoint_path=project_path(paths["vqvae_checkpoint"]),
        external_root=project_path(paths["external_sdfusion"]),
        freeze=bool(config["sdf_encoder_decoder"].get("freeze", True)),
    ).to(device)
    del encoder_decoder
    raise NotImplementedError(
        "Latent extraction needs the concrete SDF dataset reader and SDFusion VQ-VAE adapter. "
        "After those are implemented, save LatentRecord files under data/latents/."
    )


if __name__ == "__main__":
    main()
