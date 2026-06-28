from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import project_path
from src.utils.mesh import sdf_to_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PLY meshes from processed SDF .pt files.")
    parser.add_argument("--manifest", default="data/metadata/sdf_chair_train.jsonl")
    parser.add_argument("--sdf", default=None, help="Export one SDF .pt file instead of reading a manifest.")
    parser.add_argument("--output-dir", default="outputs/meshes/sdf_export")
    parser.add_argument("--limit", type=int, default=5, help="Number of meshes to export; use 0 to export all.")
    parser.add_argument("--level", type=float, default=0.0)
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(str(path))


def _load_sdf(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        sdf = payload["sdf"]
    else:
        sdf = payload
    if sdf.ndim == 4:
        sdf = sdf.squeeze(0)
    if sdf.ndim != 3:
        raise ValueError(f"Expected SDF shape [D,H,W] or [1,D,H,W], got {tuple(sdf.shape)}")
    return sdf


def _export_one(sdf_path: Path, output_path: Path, level: float) -> Path:
    sdf = _load_sdf(sdf_path).numpy()
    return sdf_to_mesh(sdf, output_path, level=level)


def main() -> None:
    args = parse_args()
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sdf is not None:
        sdf_path = _resolve_path(args.sdf)
        output_path = output_dir / f"{sdf_path.stem}.ply"
        mesh_path = _export_one(sdf_path, output_path, level=args.level)
        print(f"exported: {mesh_path}")
        return

    manifest_path = _resolve_path(args.manifest)
    exported = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if args.limit > 0 and exported >= args.limit:
                break
            record = json.loads(line)
            sdf_path = _resolve_path(record["sdf_path"])
            output_path = output_dir / f"{record['split']}_{record['model_id']}.ply"
            mesh_path = _export_one(sdf_path, output_path, level=args.level)
            print(f"exported: {mesh_path}")
            exported += 1

    print(f"done: exported {exported} meshes")


if __name__ == "__main__":
    main()
