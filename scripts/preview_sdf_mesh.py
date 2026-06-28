from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import project_path


matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export quick PNG slice previews from processed SDF .pt files.")
    parser.add_argument("--manifest", default="data/metadata/sdf_chair_train.jsonl")
    parser.add_argument("--output-dir", default="outputs/figures/sdf_preview")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_path(str(path))


def _save_slice_preview(sdf: np.ndarray, output_path: Path, title: str) -> None:
    center = tuple(size // 2 for size in sdf.shape)
    slices = [
        ("x", sdf[center[0], :, :]),
        ("y", sdf[:, center[1], :]),
        ("z", sdf[:, :, center[2]]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for axis, (name, image) in zip(axes, slices, strict=True):
        shown = axis.imshow(image.T, origin="lower", cmap="coolwarm", vmin=-0.2, vmax=0.2)
        if float(image.min()) <= 0.0 <= float(image.max()):
            axis.contour(image.T, levels=[0.0], colors="black", linewidths=0.8)
        axis.set_title(f"{name}-slice")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(shown, ax=axes, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest_path = _resolve_path(args.manifest)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if exported >= args.limit:
                break
            record = json.loads(line)
            sdf_path = _resolve_path(record["sdf_path"])
            payload = torch.load(sdf_path, map_location="cpu")
            sdf = payload["sdf"].squeeze(0).numpy()
            output_path = output_dir / f"{record['split']}_{record['model_id']}.png"
            _save_slice_preview(sdf, output_path, f"{record['split']} / {record['model_id']}")
            print(f"exported: {output_path}")
            exported += 1

    print(f"done: exported {exported} preview images")


if __name__ == "__main__":
    main()
