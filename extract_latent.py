import os
import glob
import torch
from tqdm import tqdm
from termcolor import colored
import omegaconf

# 导入底层网络骨架
from models.networks.vqvae_networks.network import VQVAE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cprint = lambda text, color: print(colored(text, color))
    
    # 1. 基础路径配置（根据实际情况微调路径）
    raw_data_dir = r"E:\PKUer\study\Frontiers_of_Geometric_Computing\geo-computing-groupwork\data\processed\sdf\test"
    output_dir = r"E:\PKUer\study\Frontiers_of_Geometric_Computing\geo-computing-groupwork\data\latents\test"
    
    config_path = "configs/vqvae_snet.yaml"
    checkpoint_path = "saved_ckpt/vqvae-snet-all.pth"
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. 拉起冷冻自编码器
    configs = omegaconf.OmegaConf.load(config_path)
    mparam = configs.model.params
    vqvae = VQVAE(ddconfig=mparam.ddconfig, n_embed=mparam.n_embed, embed_dim=mparam.embed_dim)
    vqvae.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    vqvae.to(device)
    vqvae.eval()
    
    for param in vqvae.parameters():
        param.requires_grad = False
        
    cprint("[+] VQ-VAE 生产线就位，梯度已全量锁死！", "green")

    # 3. 搜寻所有 .pt 文件
    search_pattern = os.path.join(raw_data_dir, "*.pt")
    file_list = glob.glob(search_pattern)
    
    if len(file_list) == 0:
        cprint(f"[-] 警告：在 {raw_data_dir} 下没找到任何 *.pt 文件，请检查路径！", "red")
        return
        
    cprint(f"[*] 发现可提取的真实椅子样本共计: {len(file_list)} 个。开始批量提纯...", "yellow")

    # 4. 开启流水线
    exported_count = 0
    with torch.no_grad():
        for file_path in tqdm(file_list, desc="Extracting Latents"):
            file_name = os.path.basename(file_path) # 例如: 1d9dbebc...pt
            
            # A. 专用 PyTorch 安全读取
            try:
                loaded_data = torch.load(file_path, map_location=device)
                if isinstance(loaded_data, dict) and 'sdf' in loaded_data:
                    sdf_tensor = loaded_data['sdf'].float()
                else:
                    sdf_tensor = loaded_data.float() if isinstance(loaded_data, torch.Tensor) else torch.tensor(loaded_data).float()
            except Exception as e:
                print(f"\n[-] 文件 {file_name} 读取失败，跳过。报错: {e}")
                continue

            # B. 补齐 5 维标准形状 [B=1, C=1, 64, 64, 64]
            if sdf_tensor.ndim == 3:
                sdf_tensor = sdf_tensor.unsqueeze(0).unsqueeze(0)
            elif sdf_tensor.ndim == 4:
                sdf_tensor = sdf_tensor.unsqueeze(0)
                
            sdf_tensor = sdf_tensor.to(device)

            # C. 冲入 Encoder 核心提取密码 z
            # 调用 forward_no_quant 并且只做 encode，直接拿到量化前的连续特征
            latent_z = vqvae(sdf_tensor, forward_no_quant=True, encode_only=True)
            
            # D. 剥离 Batch 维度，恢复成紧凑的 [3, 16, 16, 16] 特征矩阵
            latent_z_compact = latent_z.squeeze(0).cpu() 

            # E. 打包落盘，完好保留原文件名，方便后续队友索引
            out_path = os.path.join(output_dir, file_name)
            
            # 存成标准字典，方便一键读取：data = torch.load(path); latent = data['latent']
            payload = {
                "latent": latent_z_compact,
                "shape": list(latent_z_compact.shape) # 保存成 [3, 16, 16, 16]
            }
            torch.save(payload, out_path)
            exported_count += 1

    cprint(f"\n[###]成功将 {exported_count} 个 3D SDF 场离线压缩为轻量化特征密码盘！", "magenta")
    cprint(f"[*] 成果已悉数保存在: {output_dir}", "cyan")

if __name__ == "__main__":
    main()