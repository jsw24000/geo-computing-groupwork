"""High-level pipelines for latent extraction, DDPM training, and sampling."""

from .latent_cache import LatentRecord, load_latent_record, save_latent_record

__all__ = ["LatentRecord", "load_latent_record", "save_latent_record"]

