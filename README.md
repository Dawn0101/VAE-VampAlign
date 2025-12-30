# VAE-VampAlign: 基于 VampPrior 的变分自编码器研究

> 本项目实现了一种改进的变分自编码器（VAE），引入 **VampPrior** 和 **KL annealing** 策略，并结合深度残差网络提升生成质量。适用于 CelebA 和 MNIST 数据集的图像重建与生成任务。

---

## 📁 项目目录结构
```
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
├── results/              # 存放训练过程中生成的图像（当前已有示例图像）
├── utils/                # 工具函数
│   ├── loss.py           # 损失函数计算
│   ├── loss_curve_celebA.py
│   └── loss_curve_mnist.py
├── config.yaml           # 全局配置文件，定义数据路径、模型超参数（如 latent_dim、pseudo_inputs）、训练设置（epochs、batch_size、learning_rate）及 KL annealing 策略
├── train.py              # 主训练脚本：加载配置、初始化模型/优化器、执行训练循环、定期保存 checkpoint 和 result 结果
├── eval.py               # （定量）评估脚本：加载训练好的模型，
├── mnist_generation.py   # （定性）专门针对 MNIST 数据集的生成脚本，用于快速测试或展示模型在 MNIST 上的生成能力
└── celebA_generation.py  # （定性）专门针对 CelebA 数据集的生成脚本，用于快速测试或展示模型在 CelebA 上的生成能力
```

## 🔧 启动方式

### 训练
首先需要安装pytroch等包；

接着需要设置config.yaml的超参数，例如batch_size、lr 、input_size（分辨率，目前为64，可以修改为128）

最后在项目根目录下执行以下命令以开始训练：

```bash
python train.py

### 生成结果（定性评估）
生成mnist图像样本

```bash
python mnist_generation.py

生成celebA图像样本
```bash
python celebA_generation.py

### 评估指标（定量评估）
需要有训练好的模型权重，设置路径后运行
```bash
python eval.py

预期结果会输出Active Units和LPIPS分数。


## 预期结果展示

