from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class LatentRecord:
    latent: torch.Tensor
    category: str
    model_id: str
    sdf_path: str
    encoder_decoder_name: str
    encoder_decoder_checkpoint: str


def save_latent_record(record: LatentRecord, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(record), output_path)
    return output_path


def load_latent_record(path: str | Path, map_location: str | torch.device = "cpu") -> LatentRecord:
    payload = torch.load(path, map_location=map_location)
    return LatentRecord(**payload)
