from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from common import project_path
from src.data.binvox import read_binvox
from src.utils.config import ensure_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess meshes into SDF/TSDF grids.")
    parser.add_argument("--config", default="configs/vqvae_sdfusion.yaml")
    parser.add_argument("--category", default=None, help="Category to process. Defaults to config data.category.")
    parser.add_argument("--split", default=None, help="Dataset split to process. Defaults to config data.split.")
    parser.add_argument("--limit", type=int, default=None, help="Optional small subset size for a smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing SDF files.")
    parser.add_argument(
        "--binvox-kind",
        choices=["solid", "surface"],
        default="solid",
        help="Which ShapeNet binvox file to convert. Use solid for signed inside/outside SDF.",
    )
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(str(path))


def _manifest_path(metadata_root: Path, category: str, split: str) -> Path:
    return metadata_root / f"shapenet_{category}_{split}.jsonl"


def _load_manifest(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _binvox_path_for_mesh(mesh_path: Path, kind: str) -> Path:
    return mesh_path.with_suffix(f".{kind}.binvox")


def _resize_occupancy(occupancy: np.ndarray, resolution: int) -> np.ndarray:
    if occupancy.shape == (resolution, resolution, resolution):
        return occupancy.astype(bool)
    factors = [resolution / size for size in occupancy.shape]
    return ndimage.zoom(occupancy.astype(np.float32), zoom=factors, order=0) >= 0.5


def _occupancy_to_sdf(occupancy: np.ndarray, truncation: float) -> np.ndarray:
    if occupancy.ndim != 3:
        raise ValueError(f"Expected a 3D occupancy grid, got {occupancy.shape}")
    if not occupancy.any():
        raise ValueError("Cannot compute SDF from an empty occupancy grid")
    if occupancy.all():
        raise ValueError("Cannot compute SDF from a fully occupied grid")

    outside_distance = ndimage.distance_transform_edt(~occupancy)
    inside_distance = ndimage.distance_transform_edt(occupancy)
    voxel_size = 2.0 / max(occupancy.shape[0] - 1, 1)
    sdf = (outside_distance - inside_distance).astype(np.float32) * float(voxel_size)
    if truncation > 0:
        sdf = np.clip(sdf, -truncation, truncation)
    return sdf


def _output_path(sdf_root: Path, category: str, split: str, model_id: str) -> Path:
    return sdf_root / category / split / f"{model_id}.pt"


def _project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(project_path()))
    except ValueError:
        return str(path)


def _save_sdf(
    output_path: Path,
    sdf: np.ndarray,
    record: dict[str, str],
    binvox_path: Path,
    resolution: int,
    truncation: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sdf": torch.from_numpy(sdf).unsqueeze(0),
        "category": record["category"],
        "synset_id": record["synset_id"],
        "model_id": record["model_id"],
        "split": record["split"],
        "mesh_path": record["mesh_path"],
        "binvox_path": str(binvox_path),
        "resolution": resolution,
        "truncation": truncation,
        "sdf_convention": "negative_inside_positive_outside",
    }
    torch.save(payload, output_path)


def main() -> None:
    args = parse_args()
    config = load_config(project_path(args.config))
    paths = config["paths"]
    sdf_root = project_path(paths["sdf_root"])
    metadata_root = project_path(paths["metadata_root"])
    ensure_dirs(sdf_root, metadata_root)

    data_cfg = config["data"]
    category = args.category or str(data_cfg.get("category", "chair"))
    split = args.split or str(data_cfg.get("split", "train"))
    resolution = int(data_cfg.get("resolution", 64))
    truncation = float(data_cfg.get("truncation", 0.2))
    manifest_path = _manifest_path(metadata_root, category, split)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run scripts/prepare_shapenet_chair.py first to generate data/metadata/*.jsonl."
        )

    records = _load_manifest(manifest_path)
    if args.limit is not None:
        records = records[: args.limit]

    manifest_stem = f"sdf_{category}_{split}"
    if args.limit is not None:
        manifest_stem = f"{manifest_stem}_limit{args.limit}"
    processed_manifest = metadata_root / f"{manifest_stem}.jsonl"
    converted = 0
    skipped = 0
    failed = 0
    with processed_manifest.open("w", encoding="utf-8") as manifest_handle:
        for index, record in enumerate(records, start=1):
            model_id = record["model_id"]
            output_path = _output_path(sdf_root, category, split, model_id)
            if output_path.exists() and not args.overwrite:
                skipped += 1
            else:
                mesh_path = _resolve_path(record["mesh_path"])
                binvox_path = _binvox_path_for_mesh(mesh_path, args.binvox_kind)
                try:
                    grid = read_binvox(binvox_path)
                    occupancy = _resize_occupancy(grid.occupancy, resolution)
                    sdf = _occupancy_to_sdf(occupancy, truncation)
                    _save_sdf(output_path, sdf, record, binvox_path, resolution, truncation)
                    converted += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"[{index}/{len(records)}] failed {model_id}: {exc}")
                    continue

            manifest_payload = {
                "category": record.get("category", category),
                "synset_id": record["synset_id"],
                "model_id": model_id,
                "split": split,
                "sdf_path": _project_relative(output_path),
                "mesh_path": record["mesh_path"],
                "resolution": resolution,
                "truncation": truncation,
            }
            manifest_handle.write(json.dumps(manifest_payload, ensure_ascii=False) + "\n")
            if index == 1 or index % 100 == 0 or index == len(records):
                print(
                    f"[{index}/{len(records)}] converted={converted} "
                    f"skipped={skipped} failed={failed}"
                )

    print(f"processed manifest: {processed_manifest}")
    print(f"sdf root: {sdf_root / category / split}")
    print(f"done: converted={converted}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
