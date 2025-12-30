import torch
import torch.nn.functional as F
import numpy as np

LOG_2PI = np.log(2 * np.pi)

def log_Normal_diag(x, mean, log_var, dim=None):
    # n维高斯分布下完整的对数概率密度（含常数项）
    log_prob = -0.5 * (LOG_2PI + log_var + (x - mean)**2 / torch.exp(log_var))
    return torch.sum(log_prob, dim)

def loss_function(recon_x, x, mu, logvar, z, model, kl_weight):
    """
    Returns: Total Loss, Recon Loss, KL Loss
    """
    latent_dim = mu.size(1)
    
    # 1. Reconstruction Loss (MSE or BCE)
    # 假设输入归一化到 [0,1]，使用 MSE 
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') 
    
    # 2. KL Divergence
    # KL = log q(z|x) - log p(z) （详细证明见pdf）
    
    # A. log q(z|x) (Entropy term)
    log_q_z = log_Normal_diag(z, mu, logvar, dim=1)
    
    # B. log p(z) (Prior term)
    if model.use_vamp:
        # 获取伪输入的后验参数
        p_mu, p_logvar = model.get_p_z_params() # [K, D]
        
        # 维度扩展以进行广播计算
        # z: [Batch, 1, D]
        z_expand = z.unsqueeze(1)
        # p_mu: [1, K, D]
        p_mu_expand = p_mu.unsqueeze(0)
        p_logvar_expand = p_logvar.unsqueeze(0)
        
        # 计算 log N(z | p_mu, p_var) for all combinations
        # result: [Batch, K, D]
        log_prob = -0.5 * (np.log(2 * np.pi) + p_logvar_expand + \
                           (z_expand - p_mu_expand)**2 / torch.exp(p_logvar_expand))
        
        # Sum over latent dimension D -> [Batch, K]
        log_prob = torch.sum(log_prob, dim=2)
        
        # Log-Sum-Exp trick to calculate log sum_k (exp(log_prob))
        # log p(z) = log(1/K * sum_k N(z|uk)) = -logK + logsumexp(log_prob)
        log_p_z = torch.logsumexp(log_prob, dim=1) - np.log(model.num_pseudo)
        
    else:
        # Standard Normal Prior N(0, I)
        log_p_z = -0.5 * (np.log(2 * np.pi) + z**2).sum(dim=1)

    # Calculate KL
    kl_loss = -(log_p_z - log_q_z).sum() 
    
    # Total ELBO
    loss = recon_loss + kl_weight * kl_loss
    
    return loss, recon_loss, kl_loss