from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.view(timesteps.shape[0], *((1,) * (target.ndim - 1)))


def cosine_betas(timesteps: int, max_beta: float = 0.02) -> torch.Tensor:
    """Cosine beta schedule as proposed in 'Improved DDPM'."""
    s = 0.008
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
    f_t = torch.cos((t / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_cumprod = f_t / f_t[0]
    betas = torch.clamp(1 - alpha_cumprod[1:] / alpha_cumprod[:-1], max=max_beta)
    return betas


class GaussianDiffusion3D(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = cosine_betas(timesteps)
        else:
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

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        steps: int = 100,
        eta: float = 0.0,
        condition: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """DDIM sampling — deterministic reverse process.

        Uses the non-Markovian DDIM formulation (Song et al., 2020).
        With eta=0 this is fully deterministic; eta=1 adds DDPM-like noise.

        Key advantage over DDPM `sample()`: naturally handles non-consecutive
        timesteps without approximation error.
        """
        x = torch.randn(shape, device=device)
        schedule = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)

        for i in range(len(schedule) - 1):
            t = schedule[i]
            t_next = schedule[i + 1]
            t_tensor = t.expand(shape[0])

            predicted_noise = self.model(x, t_tensor, condition=condition)

            # alpha_cumprod[t] is the cumulative product at step t
            alpha_cumprod_t = extract(self.alpha_cumprod, t_tensor, x)
            alpha_cumprod_t_next = extract(
                self.alpha_cumprod,
                t_next.clamp(min=0).expand(shape[0]),
                x,
            )

            # sqrt with numerical protection
            sqrt_ac_t = torch.sqrt(alpha_cumprod_t.clamp(min=1e-6))
            sqrt_one_minus_ac_t = torch.sqrt((1.0 - alpha_cumprod_t).clamp(min=0.0))

            # Predict x0: (x_t - sqrt(1-ᾱₜ) * ε) / sqrt(ᾱₜ)
            x0_pred = (x - sqrt_one_minus_ac_t * predicted_noise) / sqrt_ac_t

            # DDIM step (eta=0 is fully deterministic):
            # x_{t_next} = sqrt(ᾱ_{t_next}) * x0_pred + sqrt(1-ᾱ_{t_next}) * ε
            sqrt_ac_next = torch.sqrt(alpha_cumprod_t_next.clamp(min=0.0))
            sqrt_one_minus_ac_next = torch.sqrt((1.0 - alpha_cumprod_t_next).clamp(min=0.0))

            if eta == 0.0:
                # Pure DDIM (deterministic)
                x = sqrt_ac_next * x0_pred + sqrt_one_minus_ac_next * predicted_noise
            else:
                # Stochastic DDIM with noise
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_cumprod_t_next) / (1.0 - alpha_cumprod_t).clamp(min=1e-8)
                    * (1.0 - alpha_cumprod_t / alpha_cumprod_t_next.clamp(min=1e-8))
                )
                x = sqrt_ac_next * x0_pred + torch.sqrt(sqrt_one_minus_ac_next.pow(2) - sigma.pow(2)).clamp(min=0) * predicted_noise
                x += sigma * torch.randn_like(x)

        return x
