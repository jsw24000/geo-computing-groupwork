# 基于 SDF 隐空间扩散模型的三维形状生成

本项目用于课程大作业，目标是参考 SDFusion 的思想，在 SDF 的压缩隐空间中训练扩散模型生成三维形状。当前架构不再训练自己的 VAE，而是预留接入 SDFusion 预训练 VQ-VAE 的 encoder 和 decoder。

主流程：

```text
ShapeNet / TSDF
-> SDFusion VQ-VAE encoder
-> latent cache
-> DDPM denoising training
-> SDFusion VQ-VAE decoder
-> generated SDF
-> Marching Cubes
-> mesh
```

## 当前结论

- 可以继续做去噪训练。DDPM 去噪对象是 VQ-VAE 产生的连续 latent / quantized embedding，而不是原始 SDF 网格。
- 本项目默认不扩散离散 codebook index；如果后续直接对 code index 建模，需要另做离散扩散或自回归建模。
- 不再需要 `train_vae.py`。VQ-VAE 作为冻结的外部编码器/解码器使用，项目重点转为 latent cache、DDPM 训练和采样解码。

## 外部依赖与权重

SDFusion 代码和权重作为外部资源使用，不直接混入本仓库核心源码。

- 官方仓库：[SDFusion](https://github.com/yccyenchicheng/SDFusion)
- 论文：[SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation](https://arxiv.org/abs/2212.04493)
- 推荐放置位置：`external/SDFusion/`
- 推荐 VQ-VAE 权重位置：`checkpoints/sdfusion/vqvae-snet-all.pth`

`scripts/download_sdfusion_weights.py` 只预留下载和校验入口。具体下载链接、认证和文件名以后以 SDFusion 官方 README 为准。

## 环境配置

本项目推荐使用 Conda 虚拟环境管理依赖。`.env` 更适合存路径、密钥或本机配置变量，不适合管理 PyTorch、SciPy、scikit-image 这类二进制依赖。

队友首次拿到项目后运行：

```powershell
conda env create -f environment.yml
conda activate sdf-latent-diffusion
```

验证环境：

```powershell
python -c "import torch, scipy, skimage, yaml; print(torch.__version__); print(torch.cuda.is_available())"
```

`environment.yml` 是主环境文件，面向后续 DDPM 训练和 VQ-VAE 接入，包含 PyTorch、CUDA、SciPy、scikit-image、matplotlib 等依赖。

如果只需要运行数据处理、SDF 切片预览或已有自己的 Python 环境，也可以使用 pip 备选：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

后续真正接入 SDFusion 官方代码时，如果官方仓库还有额外依赖，需要把缺失包继续合并进 `environment.yml`。

## 第一步：准备 ShapeNet chair 数据

当前阶段只需要把 ShapeNet 的 chair 类别准备好，使下一步可以交给 SDFusion 做 SDF / TSDF 预处理。ShapeNet chair 对应的 WordNet synset id 是：

```text
03001627
```

ShapeNet 数据需要通过官方授权下载，本仓库不会也不应该绕过 ShapeNet 的账号和许可。推荐流程是：

1. 从 ShapeNet 官方渠道下载并解压 `ShapeNetCore.v1`。
2. 确认解压后的目录里存在 `03001627/`。
3. 运行本项目的数据准备脚本，生成 chair manifest。

如果你已经把 ShapeNetCore.v1 解压到 `D:\datasets\ShapeNetCore.v1`，运行：

```powershell
python scripts/prepare_shapenet_chair.py --source-root D:\datasets\ShapeNetCore.v1
```

脚本会扫描：

```text
D:\datasets\ShapeNetCore.v1\03001627\<model_id>\models\model_normalized.obj
```

并生成：

```text
data/metadata/shapenet_chair_all.jsonl
data/metadata/shapenet_chair_train.jsonl
data/metadata/shapenet_chair_val.jsonl
data/metadata/shapenet_chair_test.jsonl
```

默认模式是 `manifest`，只记录原始 mesh 路径，不复制大数据。如果希望把 chair 类别复制到本项目约定目录，使用：

```powershell
python scripts/prepare_shapenet_chair.py --source-root D:\datasets\ShapeNetCore.v1 --mode copy
```

复制后目录为：

```text
data/raw/ShapeNetCore.v1/03001627/
```

如果希望节省磁盘空间，也可以尝试软链接：

```powershell
python scripts/prepare_shapenet_chair.py --source-root D:\datasets\ShapeNetCore.v1 --mode symlink
```

Windows 创建软链接可能需要开发者模式或管理员权限；如果失败，就改用默认 `manifest` 或 `copy`。

下一步接入 SDFusion 时，应让 SDFusion 的 ShapeNet 预处理脚本看到同样的类别目录：

```text
ShapeNetCore.v1/
  03001627/
    <model_id>/
      models/
        model_normalized.obj
```

本项目的 `data/metadata/*.jsonl` 用于记录我们自己的样本索引和 train/val/test 划分；SDFusion 的 SDF 生成逻辑仍以官方代码为准。

## 当前数据结构

当前本机已经将 ShapeNet chair 原始数据放入 `data/raw/`，实际目录是：

```text
data/raw/
  03001627.zip
  03001627/
    03001627/
      <model_id>/
        models/
          model_normalized.obj
          model_normalized.mtl
          model_normalized.json
          model_normalized.solid.binvox
          model_normalized.surface.binvox
```

其中 `model_normalized.obj` 是原始网格；`model_normalized.solid.binvox` 是实心 occupancy 体素，本项目当前用它计算 SDF。少数样本缺少 `solid.binvox`，预处理时会跳过。

原始样本索引位于：

```text
data/metadata/shapenet_chair_all.jsonl
data/metadata/shapenet_chair_train.jsonl
data/metadata/shapenet_chair_val.jsonl
data/metadata/shapenet_chair_test.jsonl
```

每一行记录一个样本的类别、ShapeNet synset id、model id、split、原始 mesh 路径。当前划分数量为：

```text
all:   6778
train: 6118
val:   333
test:  327
```

SDF 预处理产物放在：

```text
data/processed/sdf/train/<model_id>.pt
data/processed/sdf/val/<model_id>.pt
data/processed/sdf/test/<model_id>.pt
```

`.pt` 文件中核心字段是 `sdf`，形状为 `[1, 64, 64, 64]`，符号约定为：

```text
inside:  negative
surface: near zero
outside: positive
```

当前有效 SDF 数量为：

```text
train: 6116
val:   332
test:  327
```

对应的 SDF 索引位于：

```text
data/metadata/sdf_chair_train.jsonl
data/metadata/sdf_chair_val.jsonl
data/metadata/sdf_chair_test.jsonl
```

后续 VQ-VAE encoder 应优先读取 `sdf_chair_<split>.jsonl`，再根据其中的 `sdf_path` 加载 `.pt` 文件。

## SDF 编码/解码器包装层

`src/sdf_encoder_decoder/` 是本项目写给 SDFusion VQ-VAE 的包装层。它的目的不是复制 SDFusion 官方代码，而是把外部项目里可能比较复杂的模型构建、checkpoint 加载和函数命名，统一包装成项目内部容易使用的接口：

```python
latent = encoder_decoder.encode(sdf)
sdf = encoder_decoder.decode(latent)
```

推荐分工是：

```text
external/SDFusion/
  官方 SDFusion 源码，尽量保持原样，不在里面写本项目训练逻辑

checkpoints/sdfusion/
  官方或外部下载的 VQ-VAE 权重

src/sdf_encoder_decoder/
  本项目自己的适配代码，负责调用 external/SDFusion 并暴露统一接口
```

这样后续 DDPM 训练、采样脚本和 pipeline 都不需要关心 SDFusion 官方代码内部具体叫什么类、怎么加载配置、checkpoint 字段是什么。即使以后替换成别的 SDF encoder/decoder，也只需要新增一个同样实现 `encode()` 和 `decode()` 的包装类。

## 目录结构

```text
configs/
  vqvae_sdfusion.yaml        # SDFusion VQ-VAE 接入配置
  train_ddpm.yaml            # 无条件 latent DDPM 训练配置
  sample.yaml                # DDPM 采样和 VQ-VAE 解码配置

data/
  raw/                       # 原始 ShapeNet 或其他 mesh 数据
  processed/sdf/             # 预处理后的 SDF / TSDF 网格
  latents/                   # VQ-VAE encoder 产生的 latent cache
  metadata/                  # 样本索引、类别、路径、split 等元数据

checkpoints/
  sdfusion/                  # 外部 SDFusion VQ-VAE 权重
  ddpm/                      # 本项目训练得到的 DDPM 权重

external/
  SDFusion/                  # 可选：克隆的 SDFusion 官方代码

outputs/
  meshes/                    # 生成或重建 mesh
  figures/                   # 可视化图片
  logs/                      # 训练日志

scripts/
  download_sdfusion_weights.py
  prepare_shapenet_chair.py
  preprocess_sdf.py
  preview_sdf_mesh.py
  export_sdf_mesh.py
  encode_latents.py
  train_ddpm.py
  sample_ddpm.py
  common.py

src/
  sdf_encoder_decoder/       # SDF <-> latent 的编码/解码接口
  conditioning/              # 条件生成接口预留
  data/                      # 数据集与索引读取
  denoisers/                 # 3D U-Net 等噪声预测网络
  diffusion/                 # Gaussian DDPM 过程
  pipelines/                 # 高层训练、编码、采样流程
  utils/                     # 配置、随机种子、mesh 导出工具

milestone/                   # 里程碑报告 tex 和 pdf
AGENT.md                     # AI 编程规范和协作约定
README.md                    # 项目说明
```

## 核心接口

### SDF 编码/解码器

所有 VQ-VAE 或替代 encoder/decoder 必须实现：

```python
latent = encoder_decoder.encode(sdf)
sdf = encoder_decoder.decode(latent)
```

- `sdf`: `[B, 1, D, H, W]`
- `latent`: `[B, C, d, h, w]`
- SDFusion VQ-VAE 默认冻结参数，只做 `eval()` 推理。
- `src/sdf_encoder_decoder/base.py` 只定义抽象接口，不直接做推理。
- `src/sdf_encoder_decoder/sdfusion_vqvae.py` 是预留的 SDFusion VQ-VAE 适配器，后续真正接入官方代码时主要改这个文件。

### Denoiser

所有 DDPM denoiser 必须实现：

```python
predicted_noise = model(x_t, timesteps, condition=None)
```

- `x_t` 和 `predicted_noise` 形状一致。
- `condition=None` 是无条件生成。
- `condition=dict` 为后续类别、文本、图像、partial-shape 条件生成预留。

### Latent cache

`latent cache` 是预先算好并保存下来的 VQ-VAE latent 数据集。因为 SDFusion VQ-VAE 在本项目中是冻结的，所以没有必要在每次 DDPM 训练时重复执行：

```text
读取 SDF -> VQ-VAE encoder -> 得到 latent -> DDPM 训练
```

更推荐先单独运行一次 latent 提取流程：

```text
data/processed/sdf/ -> SDFusion VQ-VAE encoder -> data/latents/
```

之后训练 DDPM 时直接读取 `data/latents/`：

```text
latent cache -> 加噪 -> denoiser 预测噪声 -> DDPM loss
```

每个 latent 样本至少保存：

- `latent`：VQ-VAE encoder 输出的 latent 张量，形状一般为 `[C, d, h, w]` 或保存时带 batch 维。
- `category`：样本类别，例如 `chair`。
- `model_id`：原始数据集中的样本 ID，例如 ShapeNet model id。
- `sdf_path`：这个 latent 对应的预处理 SDF / TSDF 文件路径。
- `encoder_decoder_name`：生成该 latent 的编码/解码器名称，例如 `sdfusion_vqvae`。
- `encoder_decoder_checkpoint`：生成该 latent 时使用的 VQ-VAE 权重路径。

这些元数据的作用是让 DDPM 训练时知道 latent 的来源，避免后续出现“只有一个 `.pt` 文件，但不知道它来自哪个 SDF、哪个类别、哪个 VQ-VAE 权重”的问题。

## 脚本说明

### 下载或校验 SDFusion 权重

```powershell
python scripts/download_sdfusion_weights.py --target checkpoints/sdfusion/vqvae-snet-all.pth
```

当前脚本只检查目标路径并提示人工下载位置，不自动联网下载。

### 预处理 SDF

```powershell
python scripts/preprocess_sdf.py --config configs/vqvae_sdfusion.yaml --split train
python scripts/preprocess_sdf.py --config configs/vqvae_sdfusion.yaml --split val
python scripts/preprocess_sdf.py --config configs/vqvae_sdfusion.yaml --split test
```

当前脚本会读取 `data/metadata/shapenet_chair_<split>.jsonl`，找到每个样本的 `model_normalized.solid.binvox`，通过 occupancy distance transform 生成截断 SDF，并保存到 `data/processed/sdf/<split>/`。

小样本测试：

```powershell
python scripts/preprocess_sdf.py --config configs/vqvae_sdfusion.yaml --split train --limit 10
```

强制重新生成已有 SDF：

```powershell
python scripts/preprocess_sdf.py --config configs/vqvae_sdfusion.yaml --split train --overwrite
```

### 可视化 SDF

切片预览：

```powershell
python scripts/preview_sdf_mesh.py --manifest data/metadata/sdf_chair_train.jsonl --limit 5
```

输出到 `outputs/figures/sdf_preview/`。图中红色表示外部正值，蓝色表示内部负值，黑线是 `SDF=0` 的表面轮廓。

导出 Marching Cubes mesh：

```powershell
python scripts/export_sdf_mesh.py --manifest data/metadata/sdf_chair_train.jsonl --limit 5
```

输出到 `outputs/meshes/sdf_export/`。如果当前 Python 环境缺少 `skimage`，需要先按 `environment.yml` 建好 Conda 环境，或使用已经安装 `scikit-image` 的 Python 解释器运行。

### 提取 latent

```powershell
python scripts/encode_latents.py --config configs/vqvae_sdfusion.yaml
```

目标是读取 `data/processed/sdf/`，调用冻结的 SDFusion VQ-VAE encoder，并写入 `data/latents/`。

### 训练 DDPM

```powershell
python scripts/train_ddpm.py --config configs/train_ddpm.yaml
```

DDPM 只读取 latent cache 并训练 denoiser，不训练 VQ-VAE。

### 采样并解码

```powershell
python scripts/sample_ddpm.py --config configs/sample.yaml --checkpoint checkpoints/ddpm/latest.pt
```

采样 latent 后调用 SDFusion VQ-VAE decoder，最后通过 Marching Cubes 导出 mesh。

## 后续任务分工建议

- VQ-VAE 接入：实现 `src/sdf_encoder_decoder/sdfusion_vqvae.py`，适配 SDFusion 官方模型加载、`encode`、`decode`。
- 数据预处理：当前已支持从 `solid.binvox` 生成 SDF；后续如果改为从 mesh 直接计算 SDF 或复用 SDFusion 官方预处理，需要保持 `data/metadata/sdf_chair_<split>.jsonl` 的接口稳定。
- Latent cache：实现 `scripts/encode_latents.py` 或对应 pipeline，读取 `data/metadata/sdf_chair_<split>.jsonl`，保存 latent 和元数据。
- DDPM 训练：完善 `src/pipelines/train_ddpm.py`，支持断点保存、日志和 batch 训练。
- 条件生成：在 `src/conditioning/` 中逐步加入类别、文本、图像或 partial-shape 条件。

## 验证方式

重构或新增代码后先运行：

```powershell
python -m py_compile scripts/*.py src/**/*.py
```

VQ-VAE 接入后再验证：

- 单个 SDF 可以编码并保存 latent。
- 单个 latent 可以解码回 SDF / mesh。
- DDPM 可以对一个 latent batch 完成一次 loss 和 backward。

不要把接口检查结果写成最终生成质量结论。
