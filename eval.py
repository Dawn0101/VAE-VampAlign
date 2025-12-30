import torch
import torch.nn.functional as F
import lpips
import yaml
import os
from tqdm import tqdm
from models.vamp_vae import VampVAE
from data.data_loaders import CelebADataModule, MNISTDataModule

def compute_active_units(mu, threshold=0.01):
    """计算 Active Units: 方差大于 threshold 的 latent 维度数"""
    var = torch.var(mu, dim=0)
    return (var > threshold).sum().item()

def eval_model(config, ckpt_path, num_samples=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1. 加载数据（只取训练集的前 num_samples 张）===
    dataset_name = config['data']['name']
    if dataset_name == 'CelebA':
        dm = CelebADataModule(config)
        full_loader = dm.get_loader(split='train')  # 训练集
    elif dataset_name == 'MNIST':
        dm = MNISTDataModule(config)
        full_loader = dm.get_loader(train=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 提取前 num_samples 个样本
    all_data = []
    count = 0
    for batch, _ in full_loader:
        if count >= num_samples:
            break
        take = min(batch.size(0), num_samples - count)
        all_data.append(batch[:take])
        count += take
    data_tensor = torch.cat(all_data, dim=0).to(device)
    print(f"Loaded {data_tensor.size(0)} samples for evaluation.")

    # === 2. 加载模型 ===
    model = VampVAE(config).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # 初始化 LPIPS 度量
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # 使用 AlexNet 版本的 LPIPS
    total_lpips = 0.0
    all_mu = []

    with torch.no_grad():
        # 分批处理（避免显存溢出）
        batch_size = config['train'].get('batch_size', 64)
        for i in tqdm(range(0, len(data_tensor), batch_size), desc="Evaluating"):
            x = data_tensor[i:i+batch_size]

            recon, mu, logvar, z = model(x)

            # 将输入和重建图像调整到 [-1, 1] 范围，因为 LPIPS 需要这个范围的输入
            x_lpips = (x * 2) - 1
            recon_lpips = (recon * 2) - 1

            # 计算 LPIPS 损失
            lpips_loss = loss_fn_alex(x_lpips, recon_lpips).mean()
            total_lpips += lpips_loss.item()

            all_mu.append(mu)

    # 合并所有 mu
    all_mu = torch.cat(all_mu, dim=0)
    avg_lpips = total_lpips / len(data_tensor) * batch_size
    active_units = compute_active_units(all_mu)

    print(f"\n=== Evaluation Results ===")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(data_tensor)}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Active Units: {active_units} / {config['model']['latent_dim']}")
    print(f"Checkpoint: {ckpt_path}\n")

    return avg_lpips, active_units

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VAE/VampVAE using LPIPS")
    parser.add_argument('--ckpt', type=str, default="checkpoints/celebA_non_vamp_256_1000/vamp_vae_epoch_1.pth" , help='Path to model checkpoint (.pth)')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples for evaluation (default: 10000)')
    args = parser.parse_args()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Run evaluation
    lpips_score, au = eval_model(config, args.ckpt, args.num_samples)

if __name__ == '__main__':
    main()