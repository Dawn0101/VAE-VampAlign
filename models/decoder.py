from .encoder import ResBlock
import torch.nn as nn

class DecoderMnist(nn.Module):
    def __init__(self, output_channels, latent_dim):
        super(DecoderMnist, self).__init__()
        
        # 根据新的encoder输出调整全连接层
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)  # 对应新的 flatten_dim
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.Sigmoid()  # 如果输入数据归一化到了 [0,1] 范围
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 128, 4, 4)  # 确保通道数与第一层 ConvTranspose2d 的 in_channels 相匹配
        x_recon = self.net(h)
        return x_recon
    

class Decoder(nn.Module):
    def __init__(self, output_channels, latent_dim):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)  # 对应 encoder 的 flatten_dim
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 32 → 64
            nn.Sigmoid()  # 假设输入已归一化到 [0,1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)  # 通道数必须和第一层 ConvTranspose2d 的 in_channels 一致
        x_recon = self.net(h)
        return x_recon

# 改进后的 Decoder，增加了残差块以提升重建质量
class DecoderPlus(nn.Module):
    def __init__(self, output_channels, latent_dim):
        super(DecoderPlus, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.model = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 4, 4)
        return self.model(h)