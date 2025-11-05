import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取敏感性分析的CSV文件数据
batch_size_df = pd.read_csv('sensitivity_batch_size.csv')
d_model_df = pd.read_csv('sensitivity_d_model.csv')
num_layers_df = pd.read_csv('sensitivity_num_layers.csv')

# 提取数据
batch_size_values = batch_size_df['value'].tolist()
batch_size_bleu = batch_size_df['bleu'].tolist()

d_model_values = d_model_df['value'].tolist()
d_model_bleu = d_model_df['bleu'].tolist()

num_layers_values = num_layers_df['value'].tolist()
num_layers_bleu = num_layers_df['bleu'].tolist()

# 设置图形
fig, ax = plt.subplots(figsize=(14, 8))

# 计算条形图的位置
# 为每组参数设置不同的x位置，避免形状不匹配的问题
x1 = np.arange(len(batch_size_values))  # Batch Size位置
x2 = np.arange(len(d_model_values)) + len(batch_size_values) + 1  # d_model位置
x3 = np.arange(len(num_layers_values)) + len(batch_size_values) + len(d_model_values) + 2  # Num Layers位置

# 创建条形图
bars1 = ax.bar(x1, batch_size_bleu, 0.6, label='Batch Size', color='skyblue', alpha=0.8)
bars2 = ax.bar(x2, d_model_bleu, 0.6, label='d_model', color='lightgreen', alpha=0.8)
bars3 = ax.bar(x3, num_layers_bleu, 0.6, label='Num Layers', color='salmon', alpha=0.8)

# 在条形图上添加数值标签
def add_value_labels(bars, x_positions):
    for bar, x_pos in zip(bars, x_positions):
        height = bar.get_height()
        ax.text(x_pos, height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

add_value_labels(bars1, x1)
add_value_labels(bars2, x2)
add_value_labels(bars3, x3)

# 设置x轴标签
all_x = np.concatenate([x1, x2, x3])
all_labels = ([f'BS-{v}' for v in batch_size_values] + 
              [f'D-{v}' for v in d_model_values] + 
              [f'L-{v}' for v in num_layers_values])

ax.set_xlabel('Parameter Values', fontsize=12)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Sensitivity Analysis: Effect of Different Parameters on BLEU Score', fontsize=14, fontweight='bold')
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels)
ax.legend()

# 添加网格
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.xticks(rotation=45)
plt.tight_layout()

# 保存图形
plt.savefig('sensitivity_combined_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("综合敏感性分析条形图已保存为 sensitivity_combined_bar_chart.png")

# 打印一些统计信息
print("\n敏感性分析结果摘要:")
print(f"Batch Size - 最佳: {max(batch_size_bleu):.2f} (值: {batch_size_values[batch_size_bleu.index(max(batch_size_bleu))]})")
print(f"d_model - 最佳: {max(d_model_bleu):.2f} (值: {d_model_values[d_model_bleu.index(max(d_model_bleu))]})")
print(f"Num Layers - 最佳: {max(num_layers_bleu):.2f} (值: {num_layers_values[num_layers_bleu.index(max(num_layers_bleu))]})")