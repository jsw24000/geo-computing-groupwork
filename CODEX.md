# CODEX.md

## Project Context

This repository is for the course project of `几何计算前沿`. The selected topic is:

**基于 SDF 隐空间扩散模型的单类别三维形状生成**

The main route is an SDFusion-inspired but simplified pipeline:

```text
ShapeNet chair mesh
-> normalization
-> T-SDF grid
-> 3D Autoencoder / optional 3D VQ-VAE
-> latent grid
-> latent DDPM with 3D U-Net denoiser
-> decoder
-> generated SDF
-> Marching Cubes
-> generated mesh
```

The minimum goal is to run an end-to-end smoke test, not to fully reproduce a paper-level system.

## Scope

Prioritize the basic single-category unconditional generation pipeline:

1. ShapeNet chair mesh preprocessing.
2. Mesh normalization and T-SDF grid generation.
3. 3D Autoencoder training and reconstruction.
4. Latent extraction and caching.
5. Latent DDPM forward noising and denoising.
6. Minimal 3D U-Net denoiser.
7. Sampling from random latent noise.
8. Decoder reconstruction to SDF.
9. Marching Cubes mesh extraction and visualization.

Optional extensions should be treated as secondary:

1. Replace Autoencoder with VQ-VAE.
2. Add DDIM sampling.
3. Add class-conditioned generation.
4. Add text/image/partial-shape conditions.
5. Compare SDF resolution, latent resolution, and sampling settings.

Do not expand into unrelated 3D generation topics unless explicitly requested.

## Implementation Defaults

Use Python for experiments and scripts.

Prefer PyTorch for neural network modules.

Prefer simple, readable implementations over highly optimized code. The first target is correctness and reproducibility.

Use low-resolution grids first:

- Default SDF resolution: `32^3`.
- Optional higher resolution: `64^3`.
- Avoid `128^3` unless the pipeline is already stable.

Use ShapeNet `chair` as the default category. Other categories such as `airplane` or `car` are optional extensions.

Prefer a continuous 3D Autoencoder as the baseline. Implement VQ-VAE only after the AE pipeline is stable.

## Repository Organization

When creating code, prefer this structure:

```text
configs/
  smoke.yaml
  train_ae.yaml
  train_diffusion.yaml
data/
  raw/
  processed/
  latents/
outputs/
  checkpoints/
  meshes/
  figures/
src/
  data/
  models/
  diffusion/
  utils/
scripts/
  preprocess_sdf.py
  train_ae.py
  extract_latents.py
  train_diffusion.py
  sample.py
  smoke_test.py
README.md
milestone_report.md
```

Keep generated datasets, checkpoints, meshes, and figures out of source code directories.

## Reporting Style

Reports should be written in Chinese unless the user asks otherwise.

Use the following project title unless changed explicitly:

**基于 SDF 隐空间扩散模型的单类别三维形状生成**

Milestone reports should include:

1. 选题背景
2. 核心方法
3. 基础方法与可拓展方法
4. 作业进展
5. 实验与验收计划
6. 风险与备选方案

The `作业进展` section may remain partially blank until smoke tests are completed.

Avoid overstating completed work. Clearly separate:

- 已完成
- 正在实现
- 计划实现
- 可选扩展

## Smoke Test Requirements

A valid smoke test should demonstrate:

1. A small mesh or synthetic shape can be converted to an SDF/T-SDF grid.
2. The Autoencoder can run a forward pass and produce a reconstructed SDF.
3. A latent tensor can be noised and passed through the denoiser.
4. Sampling or partial denoising can produce a latent tensor of the expected shape.
5. The decoder can convert the latent to SDF.
6. Marching Cubes can export or visualize a mesh.

For smoke tests, use tiny data, few iterations, and CPU-compatible settings when possible.

## Evaluation Defaults

Qualitative evaluation:

- Visualize original mesh, reconstructed mesh, and generated mesh.
- Show several random seeds.
- Optional: latent interpolation.

Quantitative evaluation:

- Autoencoder reconstruction loss.
- Optional Chamfer Distance.
- Optional MMD / Coverage if generation quality is sufficient.

Do not make final-quality claims from smoke-test outputs.

## Risk Control

If SDF preprocessing is difficult, start with synthetic shapes or a very small mesh subset.

If VQ-VAE is unstable, fall back to the continuous Autoencoder baseline.

If 3D U-Net is too heavy, reduce grid size, latent size, channel count, batch size, or number of diffusion steps.

If generation quality is poor, first verify Autoencoder reconstruction quality before tuning diffusion.

## File Editing Rules

Do not overwrite existing reports, datasets, or logs unless explicitly requested.

Keep code comments concise and useful.

Prefer small, focused scripts that can be run independently.

When adding commands to README or reports, include the working directory assumption and expected outputs.

When writing LaTeX or Markdown, avoid claiming experiments are completed until results actually exist.

