import matplotlib.pyplot as plt

# =============== 手动提取的数据 ===============
epochs = list(range(1, 28))  # Epoch 1 to 27 (第一轮)
avg_losses = [
    55.5052, 31.0782, 28.4532, 27.3784, 26.7742,
    26.3460, 26.0322, 25.7936, 25.6203, 25.4549,
    25.3230, 25.2081, 25.1098, 25.0182, 24.9163,
    24.8680, 24.7974, 24.7208, 24.6593, 24.6199,
    24.5659, 24.5289, 24.4699, 24.4544, 24.4183,
    24.3495, 24.3  # 第27轮未完成，用第26轮值近似
]

kl_losses = [
    8.56, 9.53, 10.4, 11.2, 12.1,
    11.4, 11.6, 11.7, 11.7, 11.9,
    11.2, 10.6, 10.3, 13.0, 11.6,
    11.7, 11.9, 12.0, 11.6, 12.0,
    12.1, 13.0, 12.6, 10.7, 12.1,
    12.6, 12.8  # 第27轮用最后看到的 KL=12.8
]

# 第二轮训练（Epoch 1 to 21）
epochs2 = list(range(1, 22))
avg_losses2 = [
    47.9149, 31.4286, 29.8486, 29.0867, 28.6037,
    28.2237, 27.9748, 27.7287, 27.5705, 27.4214,
    27.2831, 27.1651, 27.0622, 26.9582, 26.8767,
    26.7929, 26.7217, 26.6680, 26.5979, 26.5515,
    26.4852
]

kl_losses2 = [
    12.1, 12.9, 12.4, 13.0, 13.0,
    13.6, 13.1, 12.3, 12.6, 12.4,
    13.7, 12.4, 13.5, 12.8, 13.6,
    13.0, 12.6, 12.7, 13.7, 13.3,
    12.3
]

# 计算 recon loss 近似值
recon_losses = [avg - kl for avg, kl in zip(avg_losses, kl_losses)]
recon_losses2 = [avg - kl for avg, kl in zip(avg_losses2, kl_losses2)]

# =============== 绘图 ===============
plt.figure(figsize=(15, 5))

# 图1: Total Loss 对比
plt.subplot(1, 3, 1)
plt.plot(epochs, avg_losses, 'b-o', label='Run 1 (with pseudo-init)')
plt.plot(epochs2, avg_losses2, 'c-s', label='Run 2 (no pseudo-init?)')
plt.xlabel('Epoch')
plt.ylabel('Average Total Loss')
plt.title('Total Loss vs Epoch')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 图2: Recon vs KL (以第一轮为例)
plt.subplot(1, 3, 2)
plt.plot(epochs, recon_losses, 'g-o', label='Reconstruction Loss')
plt.plot(epochs, kl_losses, 'r-o', label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Recon vs KL (Run 1)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 图3: KL 占比
plt.subplot(1, 3, 3)
kl_ratio = [kl / total for kl, total in zip(kl_losses, avg_losses)]
plt.plot(epochs, kl_ratio, 'm-o')
plt.xlabel('Epoch')
plt.ylabel('KL / Total Loss')
plt.title('KL Contribution Ratio')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 0.6)

plt.tight_layout()
plt.savefig('vae_mnist_training_analysis.png', dpi=200, bbox_inches='tight')
plt.show()