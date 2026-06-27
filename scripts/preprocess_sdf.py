from __future__ import annotations

import argparse

from common import project_path
from src.utils.config import ensure_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess meshes into SDF/TSDF grids.")
    parser.add_argument("--config", default="configs/vqvae_sdfusion.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    paths = config["paths"]
    ensure_dirs(project_path(paths["sdf_root"]), project_path(paths["metadata_root"]))
    raise NotImplementedError(
        "Mesh-to-SDF preprocessing is reserved for the ShapeNet integration task. "
        "Implement this script after the dataset layout is finalized."
    )


if __name__ == "__main__":
    main()

