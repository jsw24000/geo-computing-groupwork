# SDF Latent Diffusion — 3D Shape Generation

用 SDFusion 的预训练 VQ-VAE 将 SDF 压缩到 latent 空间，训练 DDPM 生成新形状。

## 数据准备

数据预处理产物：
- `data/processed/sdf/{category}/{split}/{model_id}.pt` — SDF 网格
- `data/metadata/sdf_{category}_{split}.jsonl` — 样本索引

### 提取 Latent Cache

```bash
python scripts/encode_latents.py --checkpoint saved_ckpt/vqvae-snet-all.pth --category chair --split train
python scripts/encode_latents.py --checkpoint saved_ckpt/vqvae-snet-all.pth --category chair --split val
# table, car 同理
```

输出：`data/latents/{category}/{split}/{model_id}.pt` + `stats.json`

## 训练 DDPM

每个类别独立训练一个 UNet：

```bash
python scripts/train_ddpm.py --category chair
python scripts/train_ddpm.py --category table
python scripts/train_ddpm.py --category car
```

| checkpoint存放路径 | 说明 |
|---|---|
| `checkpoints/ddpm/{category}/best.pt` | 最优权重 |
| `checkpoints/ddpm/{category}/latest.pt` | 最新权重 |
| `outputs/logs/{category}/train_log.jsonl` | 训练日志 |

## 采样

确保 **checkpoints/ddpm/{category}/best.pt** 存在后开始采样：

```bash
# DDIM 采样（推荐，200步）
python scripts/sample_ddpm.py --category car --vqvae_checkpoint saved_ckpt/vqvae-snet-all.pth --ddim --sample_steps 200 --count 10

# DDPM 采样（1000步，更精细）
python scripts/sample_ddpm.py --category car --vqvae_checkpoint saved_ckpt/vqvae-snet-all.pth --sample_steps 1000 --count 10
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--category` | — | 必填，指定类别（chair/table/car） |
| `--ddim` | 否 | 使用 DDIM（确定性，支持跳步） |
| `--sample_steps` | 100 | 采样步数（DDIM 用 200，DDPM 用 1000） |
| `--count` | 1 | 生成数量，自动递增编号 |
| `--ddpm_checkpoint` | `best.pt` | 手动指定权重路径 |
| `--device` | `auto` | 可通过 `CUDA_VISIBLE_DEVICES=""` 强制 CPU |

输出：`outputs/meshes/{category}/{category}_generated_xxx.ply`

## 断点续训

```bash
python scripts/train_ddpm.py --category chair --resume checkpoints/ddpm/chair/latest.pt
```
