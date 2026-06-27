from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ShapeNetConfig:
    root: Path
    category: str = "chair"
    resolution: int = 32
    truncation: float = 0.35


SHAPENET_SYNSET_IDS = {
    "chair": "03001627",
}


@dataclass
class ShapeNetMeshRecord:
    category: str
    synset_id: str
    model_id: str
    split: str
    mesh_path: str
    source_mesh_path: str


def category_to_synset_id(category: str) -> str:
    try:
        return SHAPENET_SYNSET_IDS[category]
    except KeyError as exc:
        known = ", ".join(sorted(SHAPENET_SYNSET_IDS))
        raise ValueError(f"Unsupported ShapeNet category '{category}'. Known categories: {known}") from exc


def resolve_category_root(source_root: str | Path, category: str = "chair") -> Path:
    source_root = Path(source_root)
    synset_id = category_to_synset_id(category)
    if source_root.name == synset_id:
        return source_root

    category_root = source_root / synset_id
    if category_root.exists():
        return category_root

    raise FileNotFoundError(
        f"Cannot find ShapeNet category folder '{synset_id}' under {source_root}. "
        "Download and extract ShapeNetCore.v1 first; chair corresponds to synset 03001627."
    )


def find_model_mesh(model_dir: Path) -> Path | None:
    preferred = [
        model_dir / "models" / "model_normalized.obj",
        model_dir / "models" / "model.obj",
        model_dir / "model_normalized.obj",
        model_dir / "model.obj",
    ]
    for path in preferred:
        if path.exists():
            return path

    for pattern in ("*.obj", "*.off", "*.ply"):
        matches = sorted(model_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def split_for_model_id(model_id: str, train_ratio: float = 0.9, val_ratio: float = 0.05) -> str:
    value = int(hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def iter_shapenet_meshes(
    source_root: str | Path,
    category: str = "chair",
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    limit: int | None = None,
) -> Iterable[ShapeNetMeshRecord]:
    category_root = resolve_category_root(source_root, category)
    synset_id = category_to_synset_id(category)
    count = 0
    for model_dir in sorted(category_root.iterdir()):
        if not model_dir.is_dir():
            continue
        mesh_path = find_model_mesh(model_dir)
        if mesh_path is None:
            continue
        model_id = model_dir.name
        yield ShapeNetMeshRecord(
            category=category,
            synset_id=synset_id,
            model_id=model_id,
            split=split_for_model_id(model_id, train_ratio=train_ratio, val_ratio=val_ratio),
            mesh_path=str(mesh_path),
            source_mesh_path=str(mesh_path),
        )
        count += 1
        if limit is not None and count >= limit:
            break


def write_manifest(records: Iterable[ShapeNetMeshRecord], metadata_dir: str | Path, stem: str) -> dict[str, Path]:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": metadata_dir / f"{stem}_all.jsonl",
        "train": metadata_dir / f"{stem}_train.jsonl",
        "val": metadata_dir / f"{stem}_val.jsonl",
        "test": metadata_dir / f"{stem}_test.jsonl",
    }
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        for record in records:
            payload = json.dumps(asdict(record), ensure_ascii=False)
            handles["all"].write(payload + "\n")
            handles[record.split].write(payload + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    return paths


def materialize_chair_subset(
    source_root: str | Path,
    target_root: str | Path,
    mode: str = "copy",
    category: str = "chair",
) -> Path:
    if mode not in {"copy", "symlink"}:
        raise ValueError("mode must be 'copy' or 'symlink'")

    category_root = resolve_category_root(source_root, category)
    synset_id = category_to_synset_id(category)
    target_category_root = Path(target_root) / synset_id
    target_category_root.parent.mkdir(parents=True, exist_ok=True)

    if target_category_root.exists():
        return target_category_root
    if mode == "copy":
        shutil.copytree(category_root, target_category_root)
    else:
        target_category_root.symlink_to(category_root, target_is_directory=True)
    return target_category_root


class ShapeNetSDFDataset:
    """Placeholder for the real ShapeNet mesh -> SDF/TSDF pipeline.

    This class marks the intended integration point for reading preprocessed
    SDF grids and metadata after the ShapeNet preprocessing pipeline exists.
    """

    def __init__(self, config: ShapeNetConfig):
        self.config = config
        self.root = Path(config.root)
        raise NotImplementedError(
            "ShapeNetSDFDataset is reserved for the real ShapeNet SDF/TSDF path. "
            "Implement it when data/processed/sdf and data/metadata layouts are finalized."
        )
