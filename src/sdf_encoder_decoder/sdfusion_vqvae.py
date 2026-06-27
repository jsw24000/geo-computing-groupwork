from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import SDFEncoderDecoder


class SDFusionVQVAEEncoderDecoder(SDFEncoderDecoder):
    """Adapter placeholder for the official SDFusion VQ-VAE.

    The official SDFusion repository uses its own model construction and
    checkpoint format. This adapter defines the project-facing API now, while
    the concrete loader can be filled in after the external repository and
    weights are available locally.
    """

    encoder_decoder_name = "sdfusion_vqvae"

    def __init__(
        self,
        checkpoint_path: str | Path,
        external_root: str | Path = "external/SDFusion",
        freeze: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.external_root = Path(external_root)
        self.model_kwargs = model_kwargs or {}
        self.model: torch.nn.Module | None = None
        if freeze:
            self.freeze()

    def load_external_model(self) -> None:
        if not self.external_root.exists():
            raise FileNotFoundError(
                f"SDFusion source tree not found: {self.external_root}. "
                "Clone the official repository into external/SDFusion first."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SDFusion VQ-VAE checkpoint not found: {self.checkpoint_path}. "
                "Download it according to the official SDFusion README."
            )
        raise NotImplementedError(
            "SDFusion model construction has not been wired yet. "
            "Adapt this method to the official SDFusion VQ-VAE module."
        )

    def encode(self, sdf: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            self.load_external_model()
        raise NotImplementedError("Wire this call to the SDFusion VQ-VAE encoder.")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            self.load_external_model()
        raise NotImplementedError("Wire this call to the SDFusion VQ-VAE decoder.")
