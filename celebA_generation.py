import torch
from torchvision import transforms
import yaml
import os
import argparse
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw, ImageFont
from models.vamp_vae import VampVAE
from data.data_loaders import CelebADataModule

def add_header_to_grid(grid_tensor, labels):
    """在顶部添加 'Original' 和 'Reconstruction' 标题"""
    ndarr = grid_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 25)
    except:
        font = ImageFont.load_default()

    w, h = img.size
    num_cols = len(labels)
    for i, label in enumerate(labels):
        section_width = w // num_cols
        text_x = i * section_width + (section_width // 2) - 40
        draw.text((text_x, 5), label, fill=(255, 255, 255), font=font)
        
    return transforms.ToTensor()(img)

def load_model(checkpoint_path, config_base, use_vamp, device):
    config = yaml.safe_load(yaml.dump(config_base)) 
    config['model']['use_vamp'] = use_vamp
    
    model = VampVAE(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Reconstruct CelebA with a single VAE model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/celebA_non_vamp_256_1000/vamp_vae_epoch_20.pth', help='Path to model checkpoint (.pth)')
    parser.add_argument('--model_type', type=str, default='std', choices=['std', 'vamp'],
                        help='Model type: "std" or "vamp"')
    parser.add_argument('--num_images', type=int, default=64, help='Number of images to reconstruct (default: 64)')
    parser.add_argument('--nrow', type=int, default=8, help='Number of images per row in the grid (default: 8)')
    parser.add_argument('--output', type=str, default='figures/reconstruction.png', help='Output image path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载配置
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    # 2. 是否使用 VampPrior
    use_vamp = (args.model_type == 'vamp')

    # 3. 加载模型
    print(f"Loading {'VampPrior' if use_vamp else 'Standard'} VAE from {args.checkpoint}...")
    model = load_model(args.checkpoint, base_config, use_vamp=use_vamp, device=device)

    # 4. 加载测试数据
    dm = CelebADataModule(base_config)
    test_loader = dm.get_loader(split='test')
    original_images, _ = next(iter(test_loader))
    original_images = original_images[:args.num_images].to(device)

    # 5. 推理重建
    print("Generating reconstructions...")
    with torch.no_grad():
        recon_images, _, _, _ = model(original_images)

    # 6. ✅ 正确拼接：先各自生成 8x8 网格，再左右拼
    grid_orig = make_grid(original_images, nrow=args.nrow, padding=2, normalize=True)
    grid_recon = make_grid(recon_images, nrow=args.nrow, padding=2, normalize=True)
    comparison_grid = torch.cat([grid_orig, grid_recon], dim=2)  # 水平拼接两个完整网格

    # 7. 添加标题
    label = "VampPrior" if use_vamp else "Standard VAE"
    # final_img = add_header_to_grid(comparison_grid, ["Original", label])

    # 8. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_image(comparison_grid, args.output)
    print(f"✅ Saved to {args.output}")

if __name__ == '__main__':
    main()