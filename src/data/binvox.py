from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BinvoxGrid:
    occupancy: np.ndarray
    dims: tuple[int, int, int]
    translate: tuple[float, float, float]
    scale: float


def read_binvox(path: str | Path) -> BinvoxGrid:
    """Read a binvox occupancy file as a boolean [X, Y, Z] grid."""

    path = Path(path)
    with path.open("rb") as handle:
        version = handle.readline().decode("ascii").strip()
        if not version.startswith("#binvox"):
            raise ValueError(f"Not a binvox file: {path}")

        dims: tuple[int, int, int] | None = None
        translate = (0.0, 0.0, 0.0)
        scale = 1.0
        while True:
            line = handle.readline().decode("ascii").strip()
            if line == "data":
                break
            if not line:
                raise ValueError(f"Unexpected end of binvox header: {path}")
            key, *values = line.split()
            if key == "dim":
                dims = tuple(int(v) for v in values)
            elif key == "translate":
                translate = tuple(float(v) for v in values)
            elif key == "scale":
                scale = float(values[0])

        if dims is None:
            raise ValueError(f"Missing binvox dimensions: {path}")

        encoded = np.frombuffer(handle.read(), dtype=np.uint8)
        if encoded.size % 2 != 0:
            raise ValueError(f"Malformed binvox RLE payload: {path}")
        values = encoded[0::2]
        counts = encoded[1::2]
        flat = np.repeat(values, counts).astype(bool)
        expected = int(np.prod(dims))
        if flat.size != expected:
            raise ValueError(f"Binvox payload has {flat.size} voxels, expected {expected}: {path}")

    # Binvox stores axes as x-z-y in the byte stream; transpose to x-y-z.
    occupancy = flat.reshape(dims).transpose(0, 2, 1)
    return BinvoxGrid(occupancy=occupancy, dims=dims, translate=translate, scale=scale)
