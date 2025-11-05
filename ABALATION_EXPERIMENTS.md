# 消融实验 (Ablation Studies)

## 概述
本项目包含多个消融实验，用于评估模型不同组件对性能的影响。消融实验通过系统地移除或修改模型的特定组件来评估其重要性。

## 实验设计

### 1. 相对位置偏置消融实验
比较包含和不包含相对位置偏置的模型性能：

- **基线模型**: 包含相对位置偏置
- **移除相对位置偏置**: 不包含相对位置偏置

### 2. 综合消融实验
更全面的消融实验，评估多个组件的影响：

- **基线模型**: 包含所有组件 (相对位置偏置, dropout=0.1, GELU, 4层)
- **移除相对位置偏置**: 不包含相对位置偏置
- **移除Dropout**: dropout=0.0
- **替换激活函数**: 使用ReLU替代GELU
- **浅层模型**: 使用2层替代4层

## 运行实验

### 运行相对位置偏置消融实验
```bash
python -m src.ablation.run_ablation_relpos
```

### 运行综合消融实验
```bash
python -m src.ablation.run_comprehensive_ablation_v2
```

## 文件结构

- `src/model_relpos.py`: 原始模型实现（包含相对位置偏置）
- `src/ablation/model_no_relpos.py`: 不包含相对位置偏置的模型变体
- `src/ablation/model_ablation.py`: 支持消融实验选项的模型实现
- `src/ablation/train_no_relpos.py`: 训练不包含相对位置偏置的模型
- `src/ablation/train_ablation.py`: 支持消融实验选项的训练脚本
- `src/ablation/run_ablation_relpos.py`: 相对位置偏置消融实验脚本
- `src/ablation/run_comprehensive_ablation_v2.py`: 综合消融实验脚本

## 结果分析

实验结果将保存在以下文件中：

- `results/ablation_comprehensive_summary.csv`: 消融实验结果汇总
- `results/ablation_comprehensive_comparison.png`: 可视化结果

## 模型组件分析

### 相对位置偏置 (Relative Position Bias)
- 作用：在注意力机制中引入相对位置信息
- 实现：类似T5的相对位置偏置表
- 影响：评估序列中元素间的相对距离对翻译质量的影响

### 激活函数 (Activation Function)
- 选项：GELU vs ReLU
- 影响：评估非线性激活函数对模型表达能力的影响

### Dropout
- 作用：正则化技术，防止过拟合
- 影响：评估正则化对模型泛化能力的影响

### 模型深度
- 选项：2层 vs 4层
- 影响：评估模型容量对性能的影响

## 评估指标

- **BLEU Score**: 评估翻译质量的主要指标
- **Validation Loss**: 训练过程中模型的损失值
- **训练时间**: 不同配置下的训练效率

## 预期结果

通过这些消融实验，我们可以：
1. 量化相对位置偏置对模型性能的贡献
2. 确定哪些组件对模型性能至关重要
3. 识别可以简化或移除的组件
4. 为模型优化提供指导

## 实验配置

默认实验参数：
- d_model: 256
- num_layers: 4 (基线)
- num_heads: 8
- d_ff: 1024
- epochs: 5 (为加快实验速度)
- batch_size: 64
- learning_rate: 3e-4

## 注意事项

- 所有实验在相同的硬件和数据集上运行以确保公平比较
- 每个实验的结果保存在独立的目录中，便于后续分析
- 实验可能需要较长时间运行，建议使用GPU加速