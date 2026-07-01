from __future__ import annotations

from pathlib import Path
from typing import Any

import omegaconf
import torch

from .base import SDFEncoderDecoder


class SDFusionVQVAEEncoderDecoder(SDFEncoderDecoder):
    """Adapter that wraps the local VQ-VAE model (models/networks/vqvae_networks).

    Builds the VQVAE from the architecture config (configs/vqvae_snet.yaml),
    loads pre-trained checkpoint weights, and exposes encode()/decode() that
    use the continuous (non-quantized) latent path for DDPM training/sampling.
    """

    encoder_decoder_name = "sdfusion_vqvae"

    def __init__(
        self,
        checkpoint_path: str | Path,
        external_root: str | Path = "external/SDFusion",
        freeze: bool = True,
        model_kwargs: dict[str, Any] | None = None,
        config_path: str | Path = "configs/vqvae_snet.yaml",
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.model_kwargs = model_kwargs or {}
        self.model: torch.nn.Module | None = None
        if freeze:
            self.freeze()

    def load_external_model(self) -> None:
        """Load the local VQVAE model with pre-trained checkpoint weights."""
        from models.networks.vqvae_networks.network import VQVAE

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"VQ-VAE checkpoint not found: {self.checkpoint_path}. "
                "Place the pre-trained vqvae-snet-all.pth at this location."
            )

        # 1. Load architecture config
        configs = omegaconf.OmegaConf.load(str(self.config_path))
        mparam = configs.model.params

        # 2. Build VQVAE
        self.model = VQVAE(
            ddconfig=mparam.ddconfig,
            n_embed=mparam.n_embed,
            embed_dim=mparam.embed_dim,
        )

        # 3. Load checkpoint (compatible with both raw state_dict and wrapped under 'vqvae' key)
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        if "vqvae" in state_dict:
            self.model.load_state_dict(state_dict["vqvae"], strict=True)
        else:
            self.model.load_state_dict(state_dict, strict=True)

        # 4. Freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def encode(self, sdf: torch.Tensor) -> torch.Tensor:
        """Encode SDF [B,1,D,H,W] → continuous latent [B,3,16,16,16]."""
        if self.model is None:
            self.load_external_model()
            self.model = self.model.to(sdf.device)
        return self.model(sdf, forward_no_quant=True, encode_only=True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent [B,3,16,16,16] → SDF [B,1,64,64,64].

        Projects the continuous latent to the nearest codebook entry first,
        then passes through post_quant_conv + decoder.  This matches how the
        VQ-VAE decoder was originally trained (it always saw quantized features).
        """
        if self.model is None:
            self.load_external_model()
            self.model = self.model.to(latent.device)
        quant, _, _ = self.model.quantize(latent, is_voxel=True)
        return self.model.decode(quant)
