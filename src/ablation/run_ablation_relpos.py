import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# 定义消融实验参数空间
ABLATION_EXPERIMENTS = {
    "with_relpos": [True, False],  # True = 使用相对位置偏置, False = 不使用相对位置偏置
}

BASE_ARGS = {
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "d_ff": 1024,
    "epochs": 5,  # 为加快实验速度，设为5轮
    "lr": 3e-4,
    "device": "cuda"
}

def run_experiment(exp_name, output_dir, use_relpos=True, **kwargs):
    """运行一次训练实验"""
    if use_relpos:
        cmd = ["python", "-m", "src.ablation.train_no_relpos"]
    else:
        cmd = ["python", "-m", "src.ablation.train_no_relpos"]
        
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--output_dir", output_dir]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Experiment {exp_name} failed.")
        return None

    # 读取BLEU分数和Loss
    try:
        bleus = np.load(os.path.join(output_dir, "bleus.npy"))
        valid_losses = np.load(os.path.join(output_dir, "valid_losses.npy"))
        final_bleu = float(bleus[-1])
        final_loss = float(valid_losses[-1])
        return {"bleu": final_bleu, "loss": final_loss}
    except Exception as e:
        print(f"[ERROR] Failed to load results for {exp_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_summary", type=str, default="results/ablation_summary.csv")
    args = parser.parse_args()

    results = []

    print("Running ablation analysis: with vs without relative position bias")
    
    # 实验1: 包含相对位置偏置
    exp_name = "with_relpos"
    output_dir = f"results/run_experiments/ablation/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    kwargs = BASE_ARGS.copy()
    metrics = run_experiment(exp_name, output_dir, use_relpos=True, **kwargs)
    if metrics:
        results.append({"model": exp_name, "has_relpos": True, "bleu": metrics["bleu"], "loss": metrics["loss"]})
    
    # 实验2: 不包含相对位置偏置
    exp_name = "without_relpos"
    output_dir = f"results/run_experiments/ablation/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    kwargs = BASE_ARGS.copy()
    metrics = run_experiment(exp_name, output_dir, use_relpos=False, **kwargs)
    if metrics:
        results.append({"model": exp_name, "has_relpos": False, "bleu": metrics["bleu"], "loss": metrics["loss"]})
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.output_summary, index=False)
    print("Ablation summary saved to:", args.output_summary)

    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # BLEU 分数对比
    plt.subplot(1, 2, 1)
    models = df["model"].tolist()
    bleu_scores = df["bleu"].tolist()
    colors = ['skyblue' if 'with' in m else 'lightcoral' for m in models]
    bars = plt.bar(models, bleu_scores, color=colors)
    plt.xlabel("Model Configuration")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score: With vs Without Relative Position Bias")
    plt.xticks(rotation=45)
    # 在柱状图上添加数值标签
    for bar, score in zip(bars, bleu_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f'{score:.2f}', ha='center', va='bottom')
    plt.grid(axis='y', alpha=0.3)
    
    # Loss 对比
    plt.subplot(1, 2, 2)
    loss_scores = df["loss"].tolist()
    bars = plt.bar(models, loss_scores, color=colors)
    plt.xlabel("Model Configuration")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss: With vs Without Relative Position Bias")
    plt.xticks(rotation=45)
    # 在柱状图上添加数值标签
    for bar, score in zip(bars, loss_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = f"results/ablation_relpos_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print("Ablation plot saved to:", plot_path)
    plt.close()
    
    print("\nAblation Analysis Results:")
    print("-" * 50)
    for idx, row in df.iterrows():
        model_type = "With Relative Position Bias" if row["has_relpos"] else "Without Relative Position Bias"
        print(f"{model_type}:")
        print(f"  BLEU Score: {row['bleu']:.4f}")
        print(f"  Validation Loss: {row['loss']:.4f}")
        print()

if __name__ == "__main__":
    main()