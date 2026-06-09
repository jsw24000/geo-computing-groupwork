# SDF Latent Diffusion Course Project

This project is a minimal implementation scaffold for the course project
`基于 SDF 隐空间扩散模型的单类别三维形状生成`.

The first runnable path is:

```text
synthetic SDF -> 3D VAE -> latent DDPM -> 3D U-Net -> VAE decoder -> SDF -> Marching Cubes -> PLY mesh
```

It intentionally does not implement VQ-VAE in the first version. SDFusion releases
VQ-VAE checkpoints, but this scaffold uses a lightweight continuous 3D VAE so that
the basic method path can run before adding codebooks or real ShapeNet data.

## Environment

Run commands from the project root:

```powershell
cd C:\Users\Lenovo\Desktop\几何计算前沿大作业
```

Required Python packages:

- `torch`
- `numpy`
- `pyyaml`
- `scikit-image`
- `matplotlib` optional

The smoke test does not require `trimesh` or ShapeNet.

## Smoke Test

```powershell
python scripts/smoke_test.py --config configs/smoke.yaml
```

Expected outputs:

- `outputs/meshes/smoke_input.ply`
- `outputs/meshes/smoke_reconstruction.ply`
- `outputs/meshes/smoke_generated.ply`
- `outputs/checkpoints/smoke_vae_unet.pt`

The generated shape is from untrained tiny models, so it only verifies the interface
and tensor flow. It should not be used as a quality result.

## Minimal Training Entrypoints

Train the lightweight VAE on synthetic SDF data:

```powershell
python scripts/train_vae.py --config configs/smoke.yaml
```

Train the latent diffusion model with the current VAE:

```powershell
python scripts/train_diffusion.py --config configs/smoke.yaml --vae-checkpoint outputs/checkpoints/vae_latest.pt
```

Sample from a smoke-test or trained checkpoint:

```powershell
python scripts/sample.py --config configs/smoke.yaml --checkpoint outputs/checkpoints/smoke_vae_unet.pt
```

## Next Steps

1. Add ShapeNet chair mesh discovery and SDF preprocessing.
2. Train the VAE long enough for meaningful reconstruction.
3. Cache VAE latents for diffusion training.
4. Train latent DDPM and compare generated meshes.
5. Optionally add VQ-VAE, DDIM, or class conditioning.

