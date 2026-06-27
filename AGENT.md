# AGENT.md

## 项目协作约定

本项目是“基于 SDF 隐空间扩散模型的三维形状生成”课程大作业。主路线以 SDFusion 的预训练 VQ-VAE 作为 SDF encoder/decoder，在其 latent 空间中训练 DDPM 去噪模型。

AI 助手和成员改代码时应优先维护这条主线：

```text
ShapeNet / TSDF
-> SDFusion VQ-VAE encoder
-> latent cache
-> DDPM denoising
-> SDFusion VQ-VAE decoder
-> SDF / mesh
```

不要重新引入自训练连续 VAE、旧 smoke test 或与主线无关的生成路线，除非团队明确决定恢复 baseline。

## 代码组织规范

- `src/sdf_encoder_decoder/` 只放 SDF 与 latent 之间的编码/解码接口和实现。
- `src/diffusion/` 只放扩散过程、噪声日程和采样逻辑。
- `src/denoisers/` 只放预测噪声的网络，例如 3D U-Net。
- `src/conditioning/` 只放条件生成相关的数据结构和编码器占位。
- `src/pipelines/` 组织高层流程，例如 latent 提取、DDPM 训练、采样解码。
- `scripts/` 只放可直接运行的命令行入口，不在脚本里堆复杂模型逻辑。
- `configs/` 保存实验配置，配置字段要和 README 中的接口约定一致。
- `data/`、`outputs/`、`checkpoints/`、`external/` 都是本地资源或生成产物区域，不应把大文件提交到仓库。

## 接口约定

所有 SDF 编码/解码器实现必须遵守：

- `encode(sdf) -> latent`
- `decode(latent) -> sdf`
- `sdf` 形状为 `[B, 1, D, H, W]`
- `latent` 形状为 `[B, C, d, h, w]`

所有 denoiser 实现必须遵守：

- `forward(x_t, timesteps, condition=None) -> predicted_noise`
- `x_t` 和 `predicted_noise` 形状完全一致
- `condition` 可以是 `None` 或字典，为后续类别、文本、图像、partial-shape 条件预留

DDPM 只训练 denoiser，不训练 SDFusion VQ-VAE。VQ-VAE 默认 `eval()`，参数冻结。

## AI 编程要求

- 修改前先读相关文件，不凭文件名猜测行为。
- 保持变更聚焦：架构调整、文档更新、实验代码不要混在一次无关改动里。
- 不要提交或覆盖数据集、权重、mesh、日志、报告 PDF，除非用户明确要求。
- 不要声称完成了未实际跑过的实验；README 和报告必须区分“已实现”“预留接口”“后续计划”。
- 新增脚本要能通过 `python -m py_compile`。
- 遇到外部依赖缺失时，用清晰错误提示说明需要安装或下载什么，不要静默失败。
- 配置示例必须能表达默认路径、输入输出和关键超参数。

## 实验记录

每次有效实验建议记录：

- 配置文件路径
- 使用的 SDFusion checkpoint
- 数据类别和样本数量
- latent 形状
- DDPM timesteps、采样步数、batch size、学习率
- 输出 checkpoint、mesh、日志路径
- 当前结果是接口验证、调参结果，还是可用于报告的正式结果

## 文件命名

- DDPM 训练入口使用 `train_ddpm.py`。
- 采样入口使用 `sample_ddpm.py`。
- latent 提取入口使用 `encode_latents.py`。
- 外部权重下载或校验入口使用 `download_sdfusion_weights.py`。
- 不再使用 `train_vae.py`、`smoke_test.py`、`smoke.yaml`。
