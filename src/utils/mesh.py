from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import measure


def _normalize_vertices(vertices: np.ndarray, resolution: int) -> np.ndarray:
    if resolution <= 1:
        return vertices
    return vertices / float(resolution - 1) * 2.0 - 1.0


def sdf_to_mesh(sdf: np.ndarray, output_path: str | Path, level: float = 0.0) -> Path:
    """Extract a mesh from an SDF grid and write it as an ASCII PLY file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sdf = np.asarray(sdf, dtype=np.float32)
    if sdf.ndim == 4:
        sdf = sdf[0]
    if sdf.ndim != 3:
        raise ValueError(f"Expected a 3D SDF grid, got shape {sdf.shape}")

    min_value = float(np.min(sdf))
    max_value = float(np.max(sdf))
    if not (min_value <= level <= max_value):
        level = 0.5 * (min_value + max_value)

    vertices, faces, _, _ = measure.marching_cubes(sdf, level=level)
    vertices = _normalize_vertices(vertices, sdf.shape[0])
    write_ply(output_path, vertices, faces)
    return output_path


def write_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")

