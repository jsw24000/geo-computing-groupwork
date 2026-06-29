import os
import argparse
import numpy as np
import torch
import omegaconf
import trimesh
from skimage import measure
from termcolor import colored

# 确保路径对齐：导入网络骨架
from models.networks.vqvae_networks.network import VQVAE

def load_vqvae_model(checkpoint_path, config_path, device):
    """读取 yaml 图纸，搭建网络并灌入预训练权重"""
    print(f"[*] 正在读取配置文件: {config_path}")
    configs = omegaconf.OmegaConf.load(config_path)
    mparam = configs.model.params
    
    vqvae = VQVAE(
        ddconfig=mparam.ddconfig,
        n_embed=mparam.n_embed,
        embed_dim=mparam.embed_dim,
    )
    
    print(f"[*] 正在加载预训练权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # 兼容处理不同的 dict 包裹版本
    if 'vqvae' in state_dict:
        vqvae.load_state_dict(state_dict['vqvae'], strict=True)
    else:
        vqvae.load_state_dict(state_dict, strict=True)
        
    vqvae = vqvae.to(device)
    vqvae.eval()
    print(colored(f"[+] VQ-VAE 模型加载成功并完全冷冻！", "green"))
    return vqvae

def sdf_to_mesh(sdf_volume, threshold=0.0):
    """利用 Marching Cubes 算法，把 3D 矩阵还原成顶点和三角面片"""
    # 规整化形状为 [64, 64, 64]
    if sdf_volume.ndim == 5:
        sdf_volume = sdf_volume[0, 0]
    elif sdf_volume.ndim == 4:
        sdf_volume = sdf_volume[0]
        
    sdf_volume_np = sdf_volume.detach().cpu().numpy()
    
    print(f"[*] 正在提取等值面。体素矩阵形状: {sdf_volume_np.shape}, 数值范围: [{sdf_volume_np.min():.3f}, {sdf_volume_np.max():.3f}]")
    
    try:
        # 调用 skimage 库的核心光栅化函数
        verts, faces, normals, values = measure.marching_cubes(
            sdf_volume_np, level=threshold
        )
        print(colored(f"[+] 成功提取网格表面: {len(verts)} 个顶点, {len(faces)} 个三角面片", "cyan"))
        return verts, faces
    except Exception as e:
        print(colored(f"[-] Marching Cubes 报错 (可能是阈值设置不对或矩阵全为空): {e}", "red"))
        return None, None

def save_mesh_as_obj(vertices, faces, output_path):
    """使用 trimesh 库封包并一键保存为标准 3D 资产 .obj"""
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_path)
        print(colored(f"[+++] 3D 资产已完美落盘: {output_path}", "green"))
        return True
    except Exception as e:
        print(colored(f"[-] 导出 .obj 失败: {e}", "red"))
        return False

def main():
    parser = argparse.ArgumentParser(description="手头清洗数据端到端重构评测")
    parser.add_argument("--npy_path", type=str, required=True, help="你手上那把椅子的 .npy 文件路径")
    parser.add_argument("--checkpoint_path", type=str, default="saved_ckpt/vqvae-snet-all.pth")
    parser.add_argument("--config_path", type=str, default="configs/vqvae_snet.yaml")
    parser.add_argument("--output_dir", type=str, default="test_vqvae_results")
    parser.add_argument("--sdf_threshold", type=float, default=0.0, help="SDF表面提取阈值，一般取 0.0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型
    vqvae = load_vqvae_model(args.checkpoint_path, args.config_path, device)

    # 2. 读取数据
    print(colored(f"\n[*] 正在读取手头清洗好的SDF文件: {args.npy_path}", "yellow"))
    
    # 1. 既然是 .pt 文件，直接用 torch.load 强行反序列化入内存
    loaded_data = torch.load(args.npy_path, map_location='cpu')
    
    # 2. 自动检查它被存成了什么格式（是纯 Tensor 还是一个带键值的字典）
    if isinstance(loaded_data, dict):
        if 'sdf' in loaded_data:
            sdf_grid = loaded_data['sdf']
            print("[*] 成功从字典中提取 ['sdf'] 张量")
        else:
            # 兼容其他可能起的键名，比如 'pc_sdf_sample' 等
            first_key = list(loaded_data.keys())[0]
            sdf_grid = loaded_data[first_key]
            print(f"[*] 成功从字典中提取第一个键值 [{first_key}] 的张量")
    else:
        sdf_grid = loaded_data

    # 3. 确保将其转化为标准的 NumPy 数组，供接下来的流程平稳运行
    if isinstance(sdf_grid, torch.Tensor):
        sdf_grid = sdf_grid.detach().cpu().numpy()
        
    print(colored(f"[+] 数据成功解析为 NumPy 数组！形状: {sdf_grid.shape}", "green"))
    # ====================================================
    
    # 3. 对齐维度至 PyTorch 标准的 5 维 [B=1, C=1, 64, 64, 64]
    original_sdf = torch.from_numpy(sdf_grid).float()
    if original_sdf.ndim == 3:
        original_sdf = original_sdf.unsqueeze(0).unsqueeze(0)
    elif original_sdf.ndim == 4:
        original_sdf = original_sdf.unsqueeze(0)
    original_sdf = original_sdf.to(device)

    # =========================================================
    # 核心验证 1: 提取并导出原版椅子的 OBJ 流形
    # =========================================================
    print(colored("\n>>> 阶段一：导出原始手头数据的 .obj 模型...", "magenta"))
    orig_verts, orig_faces = sdf_to_mesh(original_sdf, threshold=args.sdf_threshold)
    if orig_verts is not None:
        orig_obj_path = os.path.join(args.output_dir, "my_original_chair.obj")
        save_mesh_as_obj(orig_verts, orig_faces, orig_obj_path)

    # =========================================================
    # 核心验证 2: 强冲 VQ-VAE，测试压缩后能不能无损解码还原
    # =========================================================
    print(colored("\n>>> 阶段二：让数据穿梭 VQ-VAE 自编码器计算图...", "magenta"))
    with torch.no_grad():
        # 这里模拟完整的端到端重构（内部包含了 encode -> quantize -> decode）
        reconstructed_sdf, qloss = vqvae(original_sdf, verbose=False)
        print(f"[+] 穿梭完成。网络重构输出形状: {reconstructed_sdf.shape}")
        print(f"[+] 当前量化 Codebook 损失 (Quant Loss): {qloss.item():.6f}")

    # =========================================================
    # 核心验证 3: 提取并导出重构版椅子的 OBJ 流形
    # =========================================================
    print(colored("\n>>> 阶段三：导出经过 VQ-VAE 修复重建后的 .obj 模型...", "magenta"))
    recon_verts, recon_faces = sdf_to_mesh(reconstructed_sdf, threshold=args.sdf_threshold)
    if recon_verts is not None:
        recon_obj_path = os.path.join(args.output_dir, "my_reconstructed_chair.obj")
        save_mesh_as_obj(recon_verts, recon_faces, recon_obj_path)

    print(colored("\n[###] 结束！请查看 'test_vqvae_results' 文件夹下的 .obj 资产！", "yellow"))

if __name__ == "__main__":
    main()