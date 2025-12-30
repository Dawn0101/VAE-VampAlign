import torch
from torchvision import transforms
import yaml
import os
import argparse
from torchvision.utils import save_image, make_grid
from models.vamp_vae import VampVAE
from data.data_loaders import CelebADataModule
from PIL import Image, ImageDraw, ImageFont

def add_labels_to_grid(tensor_batch, labels):
    """
    在拼接好的整张网格图上方或下方添加文字标注。
    比逐个图片标注效率更高，且不容易出错。
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # 将网格张量转为 PIL 图像
    ndarr = tensor_batch.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    
    draw = ImageDraw.Draw(img)
    try:
        # 尝试加载中等大小的字体
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    # 获取图像宽度，用于平分标注位置
    w, h = img.size
    num_cols = len(labels)
    section_width = w // num_cols

    for i, label in enumerate(labels):
        # 计算文字位置 (每个区域的中心)
        # 注意：这里简单的在顶部绘制文字
        text_x = i * section_width + (section_width // 2) - 40
        draw.text((text_x, 5), label, fill=(255, 255, 255), font=font) # 白色文字
        
    return transforms.ToTensor()(img)

def main():
    parser = argparse.ArgumentParser(description="Compare Standard VAE vs VampPrior on CelebA")
    parser.add_argument('--ckpt_std', type=str, default="checkpoints/mnist_non_vamp/vamp_vae_epoch_21.pth", help='Path to standard VAE checkpoint (.pth)')
    parser.add_argument('--ckpt_vamp', type=str, default="checkpoints/mnist_vamp/vamp_vae_epoch_26.pth", help='Path to VampPrior VAE checkpoint (.pth)')
    parser.add_argument('--num_images', type=int, default=8, help='每一列显示的对比样本数')
    args = parser.parse_args()

    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    # 确保 config 中 CelebA 的 image_size 正确（如 64x64 或 128x128）
    dm = CelebADataModule(config)
    test_loader = dm.get_loader(split='test')
    
    # 获取一批测试图片
    it = iter(test_loader)
    original_images, _ = next(it)
    original_images = original_images[:args.num_images].to(device)

    # 2. 加载模型 (Standard VAE)
    config_std = config.copy()
    config_std['model']['use_vamp'] = False
    model_std = VampVAE(config_std).to(device)
    model_std.load_state_dict(torch.load(args.ckpt_std, map_location=device))
    model_std.eval()

    # 3. 加载模型 (VampPrior VAE)
    config_vamp = config.copy()
    config_vamp['model']['use_vamp'] = True
    model_vamp = VampVAE(config_vamp).to(device)
    model_vamp.load_state_dict(torch.load(args.ckpt_vamp, map_location=device))
    model_vamp.eval()

    # 4. 生成重构图
    with torch.no_grad():
        recon_std, _, _, _ = model_std(original_images)
        recon_vamp, _, _, _ = model_vamp(original_images)

    # 5. 拼接图像
    # 我们创建三列：原始图 | Standard重构 | VampPrior重构
    # 将每一组拼接成一个大网格
    # 将所有原始图排成一列，以此类推
    col1 = make_grid(original_images, nrow=1, padding=2)
    col2 = make_grid(recon_std, nrow=1, padding=2)
    col3 = make_grid(recon_vamp, nrow=1, padding=2)

    # 横向拼接三列 (C, H, W) -> 在 W 维度拼接
    combined_grid = torch.cat([col1, col2, col3], dim=2)

    # 6. 添加文字标注并保存
    # 这里的 labels 对应上面的拼接顺序
    final_image = add_labels_to_grid(combined_grid, ["Original", "Standard VAE", "VampPrior"])

    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', 'celeba_vamp_comparison.png')
    save_image(final_image, output_path)
    
    print(f"✅ 对比图已保存至: {output_path}")

if __name__ == '__main__':
    main()