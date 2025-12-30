import torch
import torch.nn as nn
import numpy as np
from .encoder import Encoder, EncoderMnist,EncoderPlus
from .decoder import Decoder, DecoderMnist,DecoderPlus

class VampVAE(nn.Module):
    def __init__(self, config):
        super(VampVAE, self).__init__()
        
        self.C = config['model']['celebA_channels']
        self.H = config['model']['input_size']
        self.W = config['model']['input_size']
        self.latent_dim = config['model']['latent_dim']
        self.use_vamp = config['model']['use_vamp']
        self.num_pseudo = config['model']['num_pseudo_inputs']

        # 实例化子模块
        self.encoder = EncoderPlus(self.C, self.latent_dim)
        self.decoder = DecoderPlus(self.C, self.latent_dim)

        # === VampPrior 核心组件 ===
        if self.use_vamp:
            # 创建 K 个伪输入 (Pseudo-inputs)
            # 形状与输入图像一致：[K, C, H, W]
            # 初始化为一个较小的值，让网络自己去学习这些“原型”
            self.pseudo_inputs = nn.Parameter(torch.rand(self.num_pseudo, self.C, self.H, self.W) * 0.01)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 1. 编码后验 q(z|x)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # 2. 解码 p(x|z)
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar, z

    def get_p_z_params(self):
        """
        计算 VampPrior 的组件参数。
        先验 p(z) = 1/K sum_k q(z | u_k)
        我们需要通过编码器运行伪输入 u_k 来得到组件的 mu 和 logvar。
        """
        if self.use_vamp:
            # 伪输入通过同一个编码器
            p_mu, p_logvar = self.encoder(self.pseudo_inputs)
            return p_mu, p_logvar
        else:
            return None, None