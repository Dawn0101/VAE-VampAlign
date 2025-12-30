import torch
import yaml
import os
from tqdm import tqdm
from models.vamp_vae import VampVAE
from data.data_loaders import CelebADataModule, MNISTDataModule
from utils.loss import loss_function
from torchvision.utils import save_image

def train(config):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories based on config paths
    results_dir = config['paths']['results_dir']
    checkpoints_dir = config['paths']['checkpoints_dir']
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Data:CelebA or MNIST  
    dm = CelebADataModule(config)
    dataloader = dm.get_loader(split='train')

    # dm = MNISTDataModule(config)
    # dataloader = dm.get_loader(train=True)

    test_data = next(iter(dataloader))[0][:8].to(device)

    # 学习率预热技巧
    target_kl_weight = config['train'].get('kl_weight', 1)

    # Model
    model = VampVAE(config).to(device)

    # Initialize Pseudo-inputs with real data if using vamp
    if config['model']['use_vamp']:
        print("Initializing pseudo-inputs with real data...")
        init_batch = next(iter(dataloader))[0]
        k = config['model']['num_pseudo_inputs']
        init_data = init_batch[:k] if init_batch.size(0) >= k else torch.cat([init_batch] * (k // init_batch.size(0) + 1))[:k]
        model.pseudo_inputs.data = init_data.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    # Training Loop
    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        # 学习率预热
        warmup_epochs = 8 
        if epoch < warmup_epochs:
            current_kl_weight = target_kl_weight * (epoch / warmup_epochs)
        else:
            current_kl_weight = target_kl_weight
        
        print(f"Epoch {epoch+1}, KL Weight: {current_kl_weight:.4f}")
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar, z = model(data)

            loss, recon, kl = loss_function(
                recon_batch, data, mu, logvar, z,
                model, target_kl_weight
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

            pbar.set_postfix({'Loss': loss.item() / len(data), 'KL': kl.item() / len(data)})

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}")

        # Visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()  # 切换到 eval 模式
                recon_test, _, _, _ = model(test_data)
                comparison = torch.cat([test_data, recon_test])
                save_image(comparison.cpu(), os.path.join(results_dir, f'recon_{epoch+1}.png'), nrow=8)

                if config['model']['use_vamp']:
                    save_image(model.pseudo_inputs[:64].cpu(),
                               os.path.join(results_dir, f'pseudo_inputs_{epoch+1}.png'),
                               nrow=8, normalize=True)
                model.train()  # 切回 train 模式

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"vamp_vae_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    # Load Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Call train function
    train(config)