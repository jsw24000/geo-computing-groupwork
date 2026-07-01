# Latent Space DDPM for SDF Shape Generation — AutoDL 执行指南

## 1. 项目结构总览（仅看和你的任务相关的文件）

```
Project/
├── configs/
│   ├── vqvae_snet.yaml              # VQ-VAE 架构参数（embed_dim=3, n_embed=8192）
│   ├── train_ddpm.yaml              # DDPM 训练配置（已修 latent_channels: 4→3）
│   └── sample.yaml                  # DDPM 采样配置（已修 latent_shape: [3,16,16,16]）
│
├── saved_ckpt/
│   └── vqvae-snet-all.pth           # ⚠️ SDFusion VQ-VAE 权重（100MB，需上传）
│
├── data/
│   ├── metadata/
│   │   ├── sdf_chair_train.jsonl     # SDF 索引（6116 训练样本）
│   │   ├── sdf_chair_val.jsonl       # SDF 索引（332 验证样本）
│   │   └── sdf_chair_test.jsonl      # SDF 索引（327 测试样本）
│   └── processed/sdf/
│       ├── train/                    # ⚠️ SDF .pt 文件 × 6116（约 7GB，需上传）
│       ├── val/                      # ⚠️ SDF .pt 文件 × 332
│       └── test/                     # ⚠️ SDF .pt 文件 × 327
│
├── models/networks/vqvae_networks/
│   ├── network.py                    # VQVAE 主模型（Encoder + Quantizer + Decoder）
│   ├── vqvae_modules.py             # 3D Encoder/Decoder 构建块
│   └── quantizer.py                  # Vector Quantizer
│
├── src/
│   ├── diffusion/
│   │   └── gaussian_diffusion.py     # GaussianDiffusion3D（DDPM 核心过程）
│   ├── denoisers/
│   │   └── unet3d.py                 # UNet3D（小型 3D U-Net，预测 noise）
│   ├── sdf_encoder_decoder/
│   │   ├── base.py                   # SDFEncoderDecoder 抽象接口
│   │   └── sdfusion_vqvae.py         # ✅ 已实现：包装本地 VQVAE 的适配器
│   ├── data/
│   │   └── latent_dataset.py         # ✅ 新建：LatentCacheDataset
│   ├── pipelines/
│   │   └── latent_cache.py           # LatentRecord 序列化
│   └── utils/
│       ├── config.py                 # YAML 配置加载
│       ├── mesh.py                   # Marching Cubes 导出 PLY
│       └── seed.py                   # 随机种子控制
│
└── scripts/
    ├── encode_latents.py             # ✅ Step1: 批量提取 latent cache
    ├── train_ddpm.py                 # ✅ Step2: DDPM 训练循环
    └── sample_ddpm.py                # ✅ Step3: 采样生成 chair mesh
```

---

## 2. DDPM 实现逻辑详解

### 整体流程

```
SDF .pt [1,64,64,64]
    ↓ VQ-VAE Encoder (freeze, 跳过 quantize)
latent cache [3,16,16,16]  ← 预先提取好
    ↓ 加噪（前向扩散）
x_t = sqrt(ᾱₜ)·x₀ + sqrt(1-ᾱₜ)·ε
    ↓ UNet3D 预测噪声 ε̂θ(x_t, t)
loss = MSE(ε̂θ, ε_true)
    ↓ optimizer.step()
    ↓（训练完成后）
随机噪声 z_T ~ N(0,I)
    ↓ 迭代去噪（reverse diffusion, 1000→100 step）
z_0（生成的 latent）[3,16,16,16]
    ↓ VQ-VAE Decoder
生成 SDF [1,64,64,64]
    ↓ Marching Cubes
chair.ply ✅
```

### 关键组件

#### GaussianDiffusion3D (`src/diffusion/gaussian_diffusion.py`)

- **噪声计划**: 线性，β ∈ [1e-4, 0.02]，T = 1000
- **前向扩散** `q_sample`: 根据 timestep t 混合 x₀ 和噪声
- **损失** `p_losses`: 加噪 → denoiser 预测噪声 → MSE
- **单步去噪** `p_sample`: DDPM 标准反向步骤
- **采样** `sample`: 从纯噪声 loop 1000 步（可调 sample_steps=100）

#### UNet3D (`src/denoisers/unet3d.py`)

UNet3D 是一个增强版 3D U-Net，用于在 latent 空间中预测噪声。

**架构（2层下采样 + 瓶颈自注意力）：**

```
输入 x [B, 3, 16, 16, 16]
    │
    ├── in_block (TimeBlock3D: 3→128)           ──→ skip1 ──→ concat ──
    ├── down1 (Conv3d k4 s2, 128→256) [16→8]                                   │
    ├── block1 (TimeBlock3D: 256→256)            ──→ skip2 ──→ concat ──        │
    │                                                                           │
    ├── down2 (Conv3d k4 s2, 256→512) [8→4]                                    │
    ├── block2 (TimeBlock3D: 512→512)                                           │
    │                                                                           │
    ├── mid1 (TimeBlock3D: 512→512)                                             │
    ├── AttnBlock3D(512)                      ← 瓶颈自注意力                    │
    ├── mid2 (TimeBlock3D: 512→512)                                             │
    │                                                                           │
    ├── up2 (ConvTranspose3d k4 s2, 512→256) [4→8]                             │
    ├── up_block2 (TimeBlock3D: 512→256)    ←── concat(skip2, up2) ──────────  │
    │                                                                           │
    ├── up1 (ConvTranspose3d k4 s2, 256→128) [8→16]                            │
    ├── up_block1 (TimeBlock3D: 256→128)    ←── concat(skip1, up1) ────────────
    │
    └── out_block (TimeBlock3D: 128→128) → Conv3d(128→3)
         │
         输出 [B, 3, 16, 16, 16]
```

**关键参数：**
- 基础通道: 64 → **128**（容量翻倍）
- 下采样层级: 1层 → **2层**（16→8→4）
- 瓶颈自注意力: 新增 **AttnBlock3D(512)** 在 4×4×4 处
- 每层 TimeBlock: 1个 → **1或2个**
- 时间编码维度: 256 → **512**
- 参数量: ~2.7M → **~78M**

**编码器-解码器路径参数：**

| 模块 | 输入→输出通道 | 空间尺寸 | 卷积参数 | 说明 |
|---|---|---|---|---|
| `in_block` | 3 → 64 | 16×16×16 | Conv3d(3→64, k=3) × 2 | 第一层残差块，含时间偏置 |
| `down` | 64 → 128 | 16→8 | Conv3d(64→128, k=4, s=2, p=1) | 4×4×4 卷积下采样 |
| `mid_block` | 128 → 128 | 8×8×8 | Conv3d(128→128, k=3) × 2 | 瓶颈层残差块 |
| `up` | 128 → 64 | 8→16 | ConvTranspose3d(128→64, k=4, s=2, p=1) | 4×4×4 转置卷积上采样 |
| `out_block` | 128 → 64 | 16×16×16 | Conv3d(128→64, k=3) × 2 | 拼接 skip 后的残差块 |
| `out` | 64 → 3 | 16×16×16 | Conv3d(64→3, k=3, p=1) | 最终投影到通道数 3 |

**参数量估算：~480K (0.48M)**

| 层 | 参数计算 | 参数量 |
|---|---|---|
| in_block: conv1 | 3×64×27 + 64 | 5,248 |
| in_block: norm1 | 2×64 | 128 |
| in_block: conv2 | 64×64×27 + 64 | 110,656 |
| in_block: norm2 | 2×64 | 128 |
| in_block: time_proj | 256×64 + 64 | 16,448 |
| in_block: skip (3→64) | 3×64×1 + 64 | 256 |
| down: conv | 64×128×64 + 128 | 524,416 |
| mid_block: conv1 | 128×128×27 + 128 | 442,496 |
| mid_block: norm1 | 2×128 | 256 |
| mid_block: conv2 | 128×128×27 + 128 | 442,496 |
| mid_block: norm2 | 2×128 | 256 |
| mid_block: time_proj | 256×128 + 128 | 32,896 |
| mid_block: skip(128→128) | Identity | 0 |
| up: conv_transpose | 128×64×64 + 64 | 524,352 |
| out_block: conv1 | 128×64×27 + 64 | 221,248 |
| out_block: norm1 | 2×64 | 128 |
| out_block: conv2 | 64×64×27 + 64 | 110,656 |
| out_block: norm2 | 2×64 | 128 |
| out_block: time_proj | 256×64 + 64 | 16,448 |
| out_block: skip(128→64) | 128×64×1 + 64 | 8,256 |
| out: conv | 64×3×27 + 3 | 5,187 |
| time_mlp | 256×512 + 512 + 512×256 + 256 | 262,912 |
| **合计** | | **~2,724,947** |

**TimeBlock3D 结构（残差块 + 时间偏置）：**

```text
输入 x [B, C_in, D, H, W]
    │
    ├── Conv3d(C_in → C_out, k=3, p=1)
    ├── GroupNorm(C_out)
    ├── SiLU
    ├── + time_bias (由 time_embedding 经 Linear 投影到 C_out 后广播)
    │
    ├── Conv3d(C_out → C_out, k=3, p=1)
    ├── GroupNorm(C_out)
    ├── SiLU
    │
    └── + skip(x)  (C_in ≠ C_out 时用 1×1×1 Conv 投影)
         │
         输出 [B, C_out, D, H, W]
```

时间嵌入 `time_bias` 是 Sinusoidal 编码 → MLP 后的 256 维向量，对每个通道学习一个偏置值，加到 GroupNorm + SiLU 后的特征图上，让 denoiser 知道当前是第几步去噪。

**时间编码流程：**

```text
timestep t (scalar)
    │
    └── SinusoidalEmbedding(dim=256)
         → [sin(t·f₀), cos(t·f₀), ..., sin(t·f₁₂₇), cos(t·f₁₂₇)]  # 256 维
         │
         └── MLP: Linear(256→512) → SiLU → Linear(512→256)
              │
              输出 time_embedding [B, 256]
              │
              ├──→ time_mlp 的输出传给每个 TimeBlock3D
              └──→ 每个 TimeBlock 用 self.time_proj 投影到 C_out 维
```

#### VQ-VAE 适配器 (`src/sdf_encoder_decoder/sdfusion_vqvae.py`)

```
encode(sdf): 调用 vqvae(sdf, forward_no_quant=True, encode_only=True)
decode(latent): 调用 vqvae.decode_no_quant(latent, force_not_quantize=True)
```

训练时 VQ-VAE **完全冻结**（eval + no_grad + requires_grad=False）

---

## 3. AutoDL 执行步骤

### 3.0 上传数据到 AutoDL

需要上传/同步到 AutoDL 实例的目录：

| 路径 | 大小 | 说明 |
|---|---|---|
| `Project/saved_ckpt/vqvae-snet-all.pth` | 100MB | VQ-VAE 权重 |
| `Project/data/processed/sdf/` | ~7GB | 预处理完毕的 SDF 文件 |
| `Project/data/metadata/` | <1MB | SDF 索引（已随仓库提交） |
| `Project/**/*.py` + `Project/**/*.yaml` | ~1MB | 你的代码仓库 |

推荐用 AutoDL 的文件上传或网盘同步。上传后目录结构应保持：

```
/root/Geometry_Calculation/Project/
├── saved_ckpt/vqvae-snet-all.pth
├── data/processed/sdf/{train,val,test}/*.pt
├── data/metadata/sdf_chair_*.jsonl
├── configs/*.yaml
├── scripts/*.py
├── src/**/*.py
└── models/**/*.py
```

### 3.1 环境配置

```bash
cd Geometry_Calculation/Project

# 创建 conda 环境（CPU 版，AutoDL 通常已预装 CUDA）
conda create -n sdf-latent-diffusion python=3.10 -y
conda activate sdf-latent-diffusion

# 安装 PyTorch（AutoDL 上请用对应 CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install numpy scipy scikit-image matplotlib PyYAML tqdm omegaconf einops
```

验证：
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 应输出 True
```

### 3.2 Step 1：提取 Latent Cache

**作用**：用冻结的 VQ-VAE 将所有 SDF 文件编码为 latent 特征，供 DDPM 训练使用。

```bash
# 提取训练集 latent
python scripts/encode_latents.py \
    --config configs/vqvae_sdfusion.yaml \
    --checkpoint saved_ckpt/vqvae-snet-all.pth \
    --split train

# 提取验证集 latent
python scripts/encode_latents.py \
    --config configs/vqvae_sdfusion.yaml \
    --checkpoint saved_ckpt/vqvae-snet-all.pth \
    --split val

# 提取测试集 latent（可选，采样时用不到）
python scripts/encode_latents.py \
    --config configs/vqvae_sdfusion.yaml \
    --checkpoint saved_ckpt/vqvae-snet-all.pth \
    --split test
```

**耗时**：GPU 上约 5-10 分钟（CPU 上 ~75 分钟）

**输出**：
```
data/latents/train/<model_id>.pt  × 6116  （每文件 ~6KB，合计 ~40MB）
data/latents/val/<model_id>.pt    × 332
data/latents/test/<model_id>.pt   × 327
```

每个 `.pt` 文件包含 `LatentRecord` 字典：
- `latent`: `[3, 16, 16, 16]` float32 张量
- `category`: "chair"
- `model_id`: ShapeNet 模型 ID
- `sdf_path`, `encoder_decoder_name`, `encoder_decoder_checkpoint`: 元数据

### 3.3 Step 2：训练 DDPM

```bash
python scripts/train_ddpm.py --config configs/train_ddpm.yaml
```

**训练配置**（`configs/train_ddpm.yaml`）：

| 参数 | 值 | 说明 |
|---|---|---|
| timesteps | 1000 | 扩散步数 |
| sample_steps | 100 | 采样步数 |
| beta_start/end | 1e-4 / 0.02 | 线性噪声计划 |
| latent_channels | 3 | 匹配实际 latent |
| **base_channels** | **128** | 增强 UNet 基础通道 |
| **time_dim** | **512** | 增强时间编码 |
| batch_size | 4 | 显存不够可调小 |
| learning_rate | 1e-4 | AdamW |
| **max_steps** | **200000** | 更大模型需更多训练 |
| save_every | 5000 | 每 N 步保存 checkpoint |

**Latent 归一化**：训练脚本自动加载 `data/latents/stats.json` 中的 mean/std，
将 latent 标准化到 N(0,1) 再训练，采样后自动反归一化还原。

**训练过程**：
```
[*] Latent normalization: mean=0.1153, std=0.3077
[*] Train samples: 6116, Val samples: 332
[*] Starting training for 200000 steps ...
Training: 100%|████████████████████| 200000/200000 [02:30:00<00:00, 22.22step/s]
  Step  5000 | train_loss: 0.0421 | val_loss: 0.0389 | best: 0.0389
  Step 10000 | train_loss: 0.0234 | val_loss: 0.0211 | best: 0.0211
  ...
[###] Training complete in 2:30:00
     Best val loss: 0.0028
```

- loss 预期从 ~1.0 开始，下降到 ~0.01~0.001
- log 文件保存在 `outputs/logs/train_log.jsonl`
- checkpoint 保存在 `checkpoints/ddpm/ddpm_step*.pt`（每 5000 步 + latest.pt）

**断点续训**：
```bash
python scripts/train_ddpm.py --config configs/train_ddpm.yaml --resume checkpoints/ddpm/latest.pt
```

**显存估算**：batch_size=4 时约 2-3GB，AutoDL 上完全没问题。

### 3.4 Step 3：采样生成 Chair

```bash
# 使用最新 checkpoint 生成（DDPM 采样）
python scripts/sample_ddpm.py \
    --config configs/sample.yaml \
    --ddpm_checkpoint checkpoints/ddpm/latest.pt \
    --vqvae_checkpoint saved_ckpt/vqvae-snet-all.pth

# 或使用 DDIM 采样（更快，确定性，推荐）
python scripts/sample_ddpm.py \
    --config configs/sample.yaml \
    --ddpm_checkpoint checkpoints/ddpm/latest.pt \
    --vqvae_checkpoint saved_ckpt/vqvae-snet-all.pth \
    --ddim --sample_steps 200
```

**采样配置**（`configs/sample.yaml`）：

| 参数 | 值 | 说明 |
|---|---|---|
| latent_shape | [3, 16, 16, 16] | 匹配实际 latent |
| sample_steps | 100 | 采样步数 |
| **base_channels** | **128** | 匹配增强 UNet |
| batch_size | 1 | 一次生成 1 个 chair |
| output_name | sample_generated.ply | 输出文件名 |

**Latent 反归一化**：脚本自动从 checkpoint 读取 `latent_stats`（mean/std），
采样完成后执行 `latent * std + mean` 还原到原始分布再解码。无需手动配置。

**输出**：`outputs/meshes/sample_generated.ply`

查看 .ply 文件可以用 MeshLab、Blender 或在线查看器。

---

## 4. 训练效果预期与调参建议

### Loss 曲线预期
- **~0-5000 steps**: loss 从 ~1.0 快速下降到 ~0.05
- **~5000-50000 steps**: loss 缓慢下降到 ~0.01
- **~50000-100000 steps**: loss 下降到 ~0.003~0.001

### 质量调参
- 增大 `max_steps` 至 200k-500k → 更好质量
- 增大 `sample_steps` 至 1000 → 更精细但更慢
- 增大 `base_channels` 至 128 → 更大的 denoiser
- 更换噪声计划（cosine 代替 linear）→ 更稳定

### 超参速查

| 参数 | 文件 | 行 | 默认值 | 建议范围 |
|---|---|---|---|---|
| `latent_channels` | `train_ddpm.yaml` | 24 | **3** | 固定（匹配 VAE） |
| `latent_shape` | `sample.yaml` | 19 | **[3,16,16,16]** | 固定 |
| **`base_channels`** | `train_ddpm.yaml` | 25 | **128** | 128（增强 UNet） |
| **`time_dim`** | `train_ddpm.yaml` | 26 | **512** | 512 |
| `batch_size` | `train_ddpm.yaml` | 13 | 4 | 4-32（看显存） |
| `learning_rate` | `train_ddpm.yaml` | 29 | 1e-4 | 1e-4 ~ 2e-4 |
| **`max_steps`** | `train_ddpm.yaml` | 30 | **200000** | 200k-500k |
| `sample_steps` | `train_ddpm.yaml` | 18 | 100 | 100-1000

---

## 5. 关键代码引用速查

| 功能 | 文件 | 行号 |
|---|---|---|
| VQVAE 编码（encode_no_quant） | `models/networks/vqvae_networks/network.py` | 84-88 |
| VQVAE 解码（decode_no_quant） | `models/networks/vqvae_networks/network.py` | 95-103 |
| VQVAE 适配器包装 | `src/sdf_encoder_decoder/sdfusion_vqvae.py` | 24-88 |
| DDPM 前向扩散 q_sample | `src/diffusion/gaussian_diffusion.py` | 42-48 |
| DDPM 训练损失 p_losses | `src/diffusion/gaussian_diffusion.py` | 50-60 |
| DDPM 采样 loop | `src/diffusion/gaussian_diffusion.py` | 82-97 |
| UNet3D denoiser | `src/denoisers/unet3d.py` | 41-79 |
| 训练循环 | `scripts/train_ddpm.py` | 73-123 |
| 采样解码脚本 | `scripts/sample_ddpm.py` | 24-89 |
| Latent 提取脚本 | `scripts/encode_latents.py` | 24-102 |
| LatentCacheDataset | `src/data/latent_dataset.py` | 10-31 |

---

## 6. 常见问题

**Q: CUDA out of memory?**
→ 减小 `batch_size`（`train_ddpm.yaml` 第 13 行）到 2 或 1

**Q: latent 提取时报 FileNotFoundError?**
→ 确认 `--checkpoint` 指向正确的 VQ-VAE 权重路径
→ 确认 `sdf_chair_*.jsonl` 中的 `sdf_path` 能找到对应的 `.pt` 文件

**Q: 生成 chair 形状很怪异？**
→ 训练步数不足，先让 loss 降到 0.01 以下
→ 增大 `sample_steps` 至 1000

**Q: VQ-VAE checkpoint 从哪里下载？**
→ 组员已经提供，在 `saved_ckpt/vqvae-snet-all.pth`
→ 或从 [SDFusion 官方仓库](https://github.com/yccyenchicheng/SDFusion) 的 Release 下载拟合所有 ShapeNet 类别的 VQ-VAE 权重

---

## 7. 完整命令拷贝区

```bash
# === 一键全流程（在 Project/ 目录下）===

# 1️⃣ 提取 latent（含统计 mean/std → data/latents/stats.json）
python scripts/encode_latents.py --config configs/vqvae_sdfusion.yaml --checkpoint saved_ckpt/vqvae-snet-all.pth --split train
python scripts/encode_latents.py --config configs/vqvae_sdfusion.yaml --checkpoint saved_ckpt/vqvae-snet-all.pth --split val

# 2️⃣ 训练 DDPM（200k 步，增强 UNet + latent 归一化）
python scripts/train_ddpm.py --config configs/train_ddpm.yaml

# 3️⃣ 采样生成（DDIM，推荐）
python scripts/sample_ddpm.py --config configs/sample.yaml --ddpm_checkpoint checkpoints/ddpm/latest.pt --vqvae_checkpoint saved_ckpt/vqvae-snet-all.pth --ddim --sample_steps 200

# 或使用 DDPM 采样
python scripts/sample_ddpm.py --config configs/sample.yaml --ddpm_checkpoint checkpoints/ddpm/latest.pt --vqvae_checkpoint saved_ckpt/vqvae-snet-all.pth --sample_steps 1000
```
