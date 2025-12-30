from matplotlib import pyplot as plt

# 修改隐空间维度为 256 （256，500）-> (256 , 1000)
celebA_256_loss_list = [274.4405, 198.7900, 189.5618, 185.4915, 182.8313, 180.7473, 179.0684, 177.7022, 176.7403, 175.9680, 175.3807, 174.8481, 174.3690, 174.0197, 173.5599, 173.3342, 172.9605, 172.6881, 172.4711, 172.2446, 171.9643]
celebA_256_kl_list = [46.8, 49.6, 52.4, 51.8, 63.8, 59.2, 58.8, 61.6, 56.0, 57.5, 55.5, 53.7, 53.9, 60.3, 62.6, 54.2, 60.5, 59.4, 60.5, 62.5, 58.8]

# non-vamp，256 latent dim, 1000 pseudo-inputs
celebA_non_256_loss_list = [518.6685, 295.6904, 257.8115, 238.7742, 227.6030, 220.0121, 214.6174, 210.4202, 207.1421, 204.2793, 202.0818, 200.2336, 198.5912, 197.1498, 195.8917, 194.6842, 193.7741, 192.6444, 191.8654, 191.0933, 190.3128, 189.6724, 188.9587, 188.3781, 187.9138]
celebA_non_256_kl_list = [62.7, 59.3, 55.5, 57.6, 56.6, 58.0, 55.2, 57.0, 56.9, 56.7, 56.8, 58.9, 61.7, 57.5, 56.1, 60.4, 57.8, 60.9, 60.2, 60.9, 60.0, 57.6, 60.3, 59.3, 59.8]


# 添加学习率预热（256，1000）+ resnet
celebA_warmlr_loss_list = [470.2531, 208.4425, 192.5420, 190.7110, 190.6121, 192.5362, 196.3185, 201.0754, 206.3201, 203.7714, 201.6729, 199.9095, 198.3291, 197.0507, 195.8018, 194.8731, 193.7969, 192.8428, 192.0327, 191.4230, 190.5113, 189.9469, 189.2342, 188.7720, 188.2229, 187.7690, 187.3347, 186.8546, 186.4428, 186.0445]
celebA_warmlr_kl_list = [5640.0, 212.0, 158.0, 122.0, 96.4, 84.7, 76.5, 68.7, 65.0, 64.4, 63.5, 62.9, 64.5, 62.3, 64.6, 63.1, 64.8, 65.2, 62.6, 64.6, 65.3, 62.2, 61.9, 63.8, 62.8, 62.0, 61.5, 63.0, 61.4, 67.5]

# 计算 KL ratio (%)
kl_ratio_vamp = [kl / loss * 100 for kl, loss in zip(celebA_256_kl_list, celebA_256_loss_list)]
kl_ratio_non = [kl / loss * 100 for kl, loss in zip(celebA_non_256_kl_list, celebA_non_256_loss_list)]

epochs_vamp = list(range(1, len(celebA_256_loss_list) + 1))
epochs_non = list(range(1, len(celebA_non_256_loss_list) + 1))

# ========== 图1: Total Loss ==========
plt.figure(figsize=(8, 5))
plt.plot(epochs_non, celebA_non_256_loss_list, label='Standard VAE (non-Vamp)', linewidth=2)
plt.plot(epochs_vamp, celebA_256_loss_list, label='VampPrior VAE', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss on CelebA (latent dim = 256)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/celeba_loss_curve.png', dpi=200)
plt.close()

# ========== 图2: KL Ratio (%)
plt.figure(figsize=(8, 5))
plt.plot(epochs_non, kl_ratio_non, label='Standard VAE (non-Vamp)', linewidth=2)
plt.plot(epochs_vamp, kl_ratio_vamp, label='VampPrior VAE', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('KL Ratio (%)')
plt.title('KL Divergence Ratio on CelebA (latent dim = 256)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/celeba_kl_ratio.png', dpi=200)
plt.close()

print("✅ Saved figures/celeba_loss_curve.png")
print("✅ Saved figures/celeba_kl_ratio.png")