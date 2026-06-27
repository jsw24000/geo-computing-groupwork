from __future__ import annotations

import argparse
from pathlib import Path

from common import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the expected local path for SDFusion VQ-VAE weights.")
    parser.add_argument("--target", default="checkpoints/sdfusion/vqvae-snet-all.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = project_path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"found SDFusion VQ-VAE checkpoint: {target}")
        return
    print(f"checkpoint not found: {target}")
    print("Download the VQ-VAE checkpoint from the official SDFusion repository:")
    print("https://github.com/yccyenchicheng/SDFusion")
    print("Then place the file at the target path above.")


if __name__ == "__main__":
    main()

