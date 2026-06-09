# Milestone 报告：基于 SDF 隐空间扩散模型的三维形状生成

## 一、选题背景

三维内容生成是几何计算与生成式模型交叉中的重要问题，目标是从随机噪声、类别条件、文本、图像或部分几何观测中生成可用的三维形状。相比二维图像生成，三维生成不仅需要保证形状的视觉合理性，还需要满足空间结构一致性、可渲染性以及后续编辑、重建和仿真的需求。

本项目计划研究一种基于 SDF latent diffusion 的三维形状生成方法。与直接在点云或高分辨率 SDF 网格上训练扩散模型相比，隐空间扩散先通过 3D Autoencoder 或 3D VQ-VAE 将三维形状压缩到较低维的 latent space，再在该隐空间中训练扩散模型，可以降低计算开销并提升生成过程的稳定性。该路线与 SDFusion 等工作思路接近，适合作为课程大作业中的研究型实现方向。

本项目的基础目标是实现一个单类别、无条件的三维形状生成流程：从 ShapeNet chair 类别中读取 mesh，将其转换为 T-SDF grid，训练可解码的三维隐空间表示，并在 latent space 中训练 DDPM，最终从随机噪声生成 SDF / mesh。

## 二、核心方法

项目整体流程分为三个阶段。

第一阶段是 SDF 数据构建。对 ShapeNet 中的 chair 类别 mesh 进行中心化和尺度归一化，将模型统一到固定空间范围内，然后在规则三维网格上计算 SDF 或 truncated SDF，得到体素形式的三维数据。初期计划优先使用 32^3 或 64^3 分辨率，以降低预处理和训练成本。

第二阶段是训练 3D shape autoencoder。基础方案采用 3D CNN Encoder-Decoder，将 T-SDF grid 编码为 latent grid，再由 decoder 重建 SDF。训练目标以 SDF 重建误差为主，可进一步加入 surface-aware loss，使模型更关注 SDF 接近 0 的表面区域。若基础 autoencoder 训练稳定，可进一步扩展为 3D VQ-VAE，引入 codebook 得到更接近 SDFusion 的离散化 latent 表示。

第三阶段是在 latent space 中训练扩散模型。对 autoencoder 编码得到的 clean latent z0 加入高斯噪声得到 zt，使用带时间步条件的 3D U-Net 预测噪声，并以噪声预测误差作为 DDPM 训练损失。生成时从标准高斯噪声 zT 出发，经过多步反向去噪得到生成 latent，再通过 decoder 解码为 SDF，最后使用 Marching Cubes 提取 mesh。

整体方法可以概括为：

```text
ShapeNet Mesh
    -> Normalization & SDF Conversion
    -> T-SDF Grid X
    -> 3D Autoencoder / VQ-VAE Encoder
    -> Latent Grid z0
    -> Forward Diffusion: z0 -> zt
    -> 3D U-Net Denoiser epsilon_theta(zt, t)
    -> Reverse Diffusion: zT -> z0
    -> Decoder
    -> Generated SDF
    -> Marching Cubes
    -> Generated 3D Mesh
```

## 三、基础方法与可拓展方法

基础方法聚焦于单类别无条件生成，具体包括 ShapeNet chair 数据预处理、T-SDF grid 构建、3D Autoencoder 训练、latent 提取与缓存、latent DDPM 训练、3D U-Net 去噪、DDPM 采样、decoder 解码以及 Marching Cubes 可视化。

可拓展方法包括以下方向：

1. 使用 3D VQ-VAE 替代普通 Autoencoder，使 latent 表示更规整，并增强与 SDFusion 类方法的关联。
2. 使用 DDIM 采样替代标准 DDPM 采样，以减少生成步骤并提升推理效率。
3. 加入类别条件，将 chair、airplane、car 等 ShapeNet 类别通过 class embedding 注入 3D U-Net，实现多类别条件生成。
4. 加入文本、图像或 partial shape 条件，通过 CLIP / CNN / shape encoder 提取条件特征，再使用 cross-attention、FiLM 或 AdaGN 注入去噪网络，扩展到多模态三维生成或形状补全任务。

## 四、作业进展

待补充。

后续可在基础代码实现并跑通 smoke test 后补充：

- 已完成的基础模块：
- 当前使用的数据集或测试样例：
- smoke test 结果：
- 当前可视化结果：
- 后续计划：

## 五、实验与验收计划

基础验收目标是跑通端到端 smoke test：读取少量 ShapeNet chair mesh，完成归一化和 T-SDF 转换，训练或加载一个最小规模 3D Autoencoder，并验证 SDF 能够被重建和 Marching Cubes 可视化。随后在 autoencoder latent 上进行一次 DDPM 训练或推理测试，验证加噪、去噪、采样、解码流程能够正确运行。

定性评估将展示真实样本、autoencoder 重建样本和扩散生成样本的 mesh 可视化结果，并观察形状结构是否合理。若时间允许，将进一步展示不同随机种子下的生成多样性以及 latent interpolation 的平滑过渡效果。

定量评估方面，autoencoder 阶段可使用 SDF reconstruction loss 或 Chamfer Distance；生成阶段可尝试使用 Chamfer Distance、Minimum Matching Distance、Coverage 等指标，对生成样本与真实样本集合进行比较。

## 六、风险与备选方案

SDF 预处理和 3D 网络训练计算量较大，因此初期优先使用 32^3 或 64^3 低分辨率 T-SDF，并只选取 ShapeNet chair 单类别进行实验。

若 VQ-VAE 训练不稳定，则先完成普通 3D Autoencoder 版本，确保 latent space 可解码，再将 VQ codebook 作为扩展项加入。

若 3D U-Net 显存压力较大，则降低 latent 分辨率和网络通道数，并优先完成较小步数的 smoke test。若完整 DDPM 采样耗时较长，则将 DDIM 加速采样作为后续优化方向。

本项目的最低完成标准是实现 SDF latent diffusion 的基础流程并获得可视化结果；更进一步的目标是在此基础上完成 VQ-VAE、多类别条件生成或更高效采样策略中的至少一种扩展。
