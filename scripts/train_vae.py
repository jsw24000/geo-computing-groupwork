from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from common import project_path
from src.data.synthetic_sdf import SyntheticSDFConfig, SyntheticSDFDataset
from src.models.vae import SDFVAE
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight 3D SDF VAE on synthetic SDF data.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    ensure_dirs(config["outputs"]["checkpoints"])

    data_cfg = config["data"]
    vae_cfg = config["vae"]
    dataset = SyntheticSDFDataset(
        SyntheticSDFConfig(
            resolution=int(data_cfg["resolution"]),
            num_samples=int(data_cfg["num_samples"]),
            shape_types=tuple(data_cfg["shape_types"]),
            truncation=float(data_cfg["truncation"]),
        )
    )
    loader = DataLoader(dataset, batch_size=int(data_cfg["batch_size"]), shuffle=True)

    model = SDFVAE(
        in_channels=int(vae_cfg["in_channels"]),
        base_channels=int(vae_cfg["base_channels"]),
        latent_channels=int(vae_cfg["latent_channels"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(vae_cfg["learning_rate"]))

    model.train()
    last_loss = None
    for step, batch in zip(range(int(vae_cfg["train_steps"])), loader):
        sdf = batch["sdf"].to(device)
        output = model(sdf)
        losses = model.loss(sdf, output, kl_weight=float(vae_cfg["kl_weight"]))
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        last_loss = float(losses["loss"].detach())
        print(f"step {step + 1}: loss={last_loss:.6f}")

    checkpoint_path = project_path(config["outputs"]["checkpoints"], "vae_latest.pt")
    torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
    print(f"saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
