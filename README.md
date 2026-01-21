# MSCA-Battery-Life-Prediction
论文《基于MSCA插件的锂电池寿命预测方法》的官方PyTorch实现。(MSCA: 多尺度循环注意力模块)
# MSCA: 一种用于电池寿命预测的即插即用多尺度循环注意力模块

本仓库是论文 **《基于MSCA插件的锂电池寿命预测方法》** 的官方PyTorch实现。

我们提出的 **MSCA (Multi-Scale Cyclic Attention)** 模块是一种即插即用的特征增强插件，旨在显著提升多种深度学习模型在电池剩余使用寿命（RUL）预测任务上的性能。

<p align="center">
  <img width="472" height="645" alt="image" src="https://github.com/user-attachments/assets/834b66c3-2ea6-48a3-8395-c8c13fe83073" />
  <br>
  MSCA框架图
</p>

---

## 🌟 项目亮点

- **即插即用**: 可轻松将MSCA模块集成到多种现有的骨干模型（如MLP, LSTM, GRU, Transformer等）中，无需大规模修改即可提升其性能。
- **多尺度特征提取**: 利用双路径架构（多尺度卷积 + 线性路径），从原始的循环曲线（V-Q-dQ/dV）中同时捕捉局部形态模式和全局分布特征。
- **先进的融合与精炼**: 采用自适应门控机制进行智能特征融合，并利用循环内注意力机制对特征进行二次精炼。
- **领先的性能**: 在一个包含12个公开数据集的大规模、高多样性基准上取得了优异的预测性能。

---

## 🚀 快速开始

### 1. 环境准备
请确保您的环境满足要求，并安装必要的依赖库：

```bash
pip install -r requirements.txt
```

### 2. 模型训练与评估
您可以直接运行 run_main.py 脚本进行训练。具体的参数配置（如层数、序列长度等）可以在代码中进行调整，或通过命令行参数指定。

---

## 📈 主要结果

我们的MSCA-MLP模型在综合基准测试中取得了领先的性能。

| 模型 | 15%-Acc (%) | 10%-Acc (%) |
| :--- | :--- | :--- |
| MLP | 59.73 | 47.25 |
| MSCA-MLP (本文方法) | 65.51 | 50.34 |

更详细的实验结果，包括消融实验和与其他骨干网络的结合效果，请参阅我们的论文。

---

## 🙏 致谢

本项工作的完成离不开以下优秀的开源项目，在此向原作者表示感谢：

- [BatteryLife](https://github.com/Ruifeng-Tan/BatteryLife)
- [BatteryML](https://github.com/microsoft/BatteryML)
