from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.view(timesteps.shape[0], *((1,) * (target.ndim - 1)))


class GaussianDiffusion3D(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_start) * x_start
            + extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start) * noise
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        condition: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, timesteps, noise)
        predicted_noise = self.model(x_noisy, timesteps, condition=condition)
        loss = F.mse_loss(predicted_noise, noise)
        return {"loss": loss, "predicted_noise": predicted_noise, "noise": noise, "x_noisy": x_noisy}

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        betas_t = extract(self.betas, timesteps, x)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, timesteps, x)

        predicted_noise = self.model(x, timesteps, condition=condition)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        posterior_variance_t = extract(self.posterior_variance, timesteps, x)

        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().view(x.shape[0], *((1,) * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        steps: int | None = None,
        condition: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        if steps is None or steps >= self.timesteps:
            schedule = range(self.timesteps - 1, -1, -1)
        else:
            schedule = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long).tolist()
        for timestep in schedule:
            t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            x = self.p_sample(x, t, condition=condition)
        return x
