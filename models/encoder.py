# /models/encoder.py 将输入张量 [B , channels, H, W] 编码为潜在变量的均值和对数方差
# （通用模块）适用于 VampVAE 和标准 VAE

import torch.nn as nn

class EncoderMnist(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(EncoderMnist, self).__init__()
        
        # Input: (B, input_channels, 32, 32)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),             # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # 4x4
            nn.ReLU(),
            # 注意：不再需要第4层（原为64→32→16→8→4，现在32→16→8→4）
        )

        self.flatten_dim = 128 * 4 * 4  # 因为最后一层输出通道是128，尺寸是4x4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.conv_layers(x)                # [B, 128, 4, 4]
        h = h.view(h.size(0), -1)              # [B, 128*4*4]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        
        # Input: (B, input_channels, 64, 64)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),             # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # 4x4
            nn.ReLU(),

            # celebA 输入尺寸 64x64 时新增一层
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),            # 4x4 ← 新增一层
            nn.ReLU(),
        )

        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.conv_layers(x)                # [B, 256, 4, 4]
        h = h.view(h.size(0), -1)              # [B, 256*4*4]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x)) # 跳跃连接
    
# 改进后的 Encoder，增加了残差块以提升特征提取能力
class EncoderPlus(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(EncoderPlus, self).__init__()
        
        self.model = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResBlock(64), # 增加特征提取能力

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResBlock(128),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResBlock(256),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.model(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)