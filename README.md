# VAE-VampAlign: 基于 VampPrior 的变分自编码器研究

> 本项目实现了一种改进的变分自编码器（VAE），引入 **VampPrior** 和 **KL annealing** 策略，并结合深度残差网络提升生成质量。适用于 CelebA 和 MNIST 数据集的图像重建与生成任务。

---

## 📁 项目目录结构

VAE-VampAlign/
├── data/                 # 数据加载相关
│   └── data_loaders.py
├── figures/              # 训练结果图、生成样本图
│   ├── celeba_kl_ratio.png
│   ├── celeba_loss_curve.png
│   ├── reconstruction.png
│   └── vae_mnist_training_analysis.png
├── models/               # 模型定义
│   ├── encoder.py        # 编码器（ResNet-based）
│   ├── decoder.py        # 解码器（ResNet-based）
│   └── vamp_vae.py       # VampPrior-VAE 主模型
├── results/              # 存放训练过程中生成的图像和检查点
├── utils/                # 工具函数
│   ├── loss.py           # 损失函数计算
│   ├── loss_curve_celebA.py
│   └── loss_curve_mnist.py


## 🔧 启动方式

### 训练

请在项目根目录下执行以下命令以开始训练：

```bash
python train.py --dataset [celeba|mnist] --epochs [epoch_number]