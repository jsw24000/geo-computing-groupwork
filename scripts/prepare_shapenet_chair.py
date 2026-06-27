from __future__ import annotations

import argparse
from pathlib import Path

from common import project_path
from src.data.shapenet import (
    category_to_synset_id,
    iter_shapenet_meshes,
    materialize_chair_subset,
    resolve_category_root,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the ShapeNetCore.v1 chair category for later SDFusion preprocessing. "
            "This script expects ShapeNet to have already been downloaded with official access."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Extracted ShapeNetCore.v1 root, or the chair synset folder itself.",
    )
    parser.add_argument("--target-root", default="data/raw/ShapeNetCore.v1")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--category", default="chair")
    parser.add_argument("--mode", choices=["manifest", "copy", "symlink"], default="manifest")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None, help="Optional small subset size for local checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser()
    target_root = project_path(args.target_root)
    metadata_dir = project_path(args.metadata_dir)
    synset_id = category_to_synset_id(args.category)

    records = list(
        iter_shapenet_meshes(
            source_root=source_root,
            category=args.category,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            limit=args.limit,
        )
    )
    if not records:
        raise RuntimeError(
            f"No mesh files found for {args.category} ({synset_id}) under {source_root}. "
            "Expected files such as 03001627/<model_id>/models/model_normalized.obj."
        )

    source_category_root = resolve_category_root(source_root, args.category).resolve()
    if args.mode in {"copy", "symlink"}:
        prepared_category_root = materialize_chair_subset(
            source_root=source_root,
            target_root=target_root,
            mode=args.mode,
            category=args.category,
        )
        prepared_prefix = (target_root / synset_id).resolve()
        updated_records = []
        for record in records:
            mesh_path = Path(record.mesh_path).resolve()
            try:
                relative_path = mesh_path.relative_to(source_category_root)
            except ValueError:
                relative_path = Path(record.model_id) / "models" / mesh_path.name
            record.mesh_path = str(prepared_prefix / relative_path)
            updated_records.append(record)
        records = updated_records
    else:
        prepared_category_root = source_category_root

    manifest_paths = write_manifest(records, metadata_dir, stem=f"shapenet_{args.category}")
    split_counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        split_counts[record.split] += 1

    print(f"category: {args.category}")
    print(f"synset id: {synset_id}")
    print(f"mesh count: {len(records)}")
    print(f"prepared category root: {prepared_category_root}")
    print(f"split counts: {split_counts}")
    for name, path in manifest_paths.items():
        print(f"{name} manifest: {path}")
    print("Next: run SDFusion's ShapeNet SDF preprocessing against the prepared category folder.")


if __name__ == "__main__":
    main()
