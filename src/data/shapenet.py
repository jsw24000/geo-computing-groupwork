from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShapeNetConfig:
    root: Path
    category: str = "chair"
    resolution: int = 32
    truncation: float = 0.35


class ShapeNetSDFDataset:
    """Placeholder for the real ShapeNet mesh -> T-SDF pipeline.

    The first project version uses synthetic SDFs for smoke tests because the
    local environment does not currently provide ShapeNet meshes or trimesh.
    This class marks the intended integration point for the next milestone.
    """

    def __init__(self, config: ShapeNetConfig):
        self.config = config
        self.root = Path(config.root)
        raise NotImplementedError(
            "ShapeNetSDFDataset is reserved for the real ShapeNet preprocessing path. "
            "Use SyntheticSDFDataset for the current smoke-test implementation."
        )

