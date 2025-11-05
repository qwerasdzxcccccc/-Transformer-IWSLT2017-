import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# 定义消融实验配置
ABLATION_EXPERIMENTS = {
    "baseline": {
        "use_relpos": True,
        "dropout": 0.1,
        "use_gelu": True,
        "num_layers": 4
    },
    "no_relpos": {
        "use_relpos": False,
        "dropout": 0.1,
        "use_gelu": True,
        "num_layers": 4
    },
    "no_dropout": {
        "use_relpos": True,
        "dropout": 0.0,
        "use_gelu": True,
        "num_layers": 4
    },
    "no_gelu": {
        "use_relpos": True,
        "dropout": 0.1,
        "use_gelu": False,
        "num_layers": 4
    },
    "shallow": {
        "use_relpos": True,
        "dropout": 0.1,
        "use_gelu": True,
        "num_layers": 2
    }
}

BASE_ARGS = {
    "d_model": 256,
    "num_heads": 8,
    "d_ff": 1024,
    "epochs": 5,
    "lr": 3e-4,
    "device": "cuda",
    "batch_size": 64
}

def run_experiment(exp_name, output_dir, **kwargs):
    """运行一次训练实验"""
    cmd = ["python", "-m", "src.ablation.train_ablation"]
        
    # 添加所有参数
    for k, v in kwargs.items():
        # 将布尔值转换为字符串
        if isinstance(v, bool):
            cmd += [f"--{k}", str(v).lower()]
        else:
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
    parser.add_argument("--output_summary", type=str, default="results/ablation_comprehensive_summary.csv")
    args = parser.parse_args()

    results = []

    print("Running comprehensive ablation analysis...")
    
    for exp_name, exp_params in ABLATION_EXPERIMENTS.items():
        print(f"\nRunning experiment: {exp_name}")
        output_dir = f"results/run_experiments/ablation_comprehensive/{exp_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 合并基础参数和实验特定参数
        kwargs = BASE_ARGS.copy()
        kwargs.update(exp_params)
        
        metrics = run_experiment(exp_name, output_dir, **kwargs)
        if metrics:
            results.append({
                "experiment": exp_name, 
                "bleu": metrics["bleu"], 
                "loss": metrics["loss"],
                "params": exp_params
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.output_summary, index=False)
    print("\nAblation summary saved to:", args.output_summary)

    # 可视化结果
    plt.figure(figsize=(15, 6))
    
    # BLEU 分数对比
    plt.subplot(1, 2, 1)
    exp_names = df["experiment"].tolist()
    bleu_scores = df["bleu"].tolist()
    
    # 为不同的实验设置不同颜色
    colors = []
    for name in exp_names:
        if name == "baseline":
            colors.append('gold')
        elif "no_relpos" in name:
            colors.append('lightcoral')
        elif "no_dropout" in name:
            colors.append('lightblue')
        elif "no_gelu" in name:
            colors.append('plum')
        elif "shallow" in name:
            colors.append('lightgreen')
        else:
            colors.append('lightgray')
    
    bars = plt.bar(exp_names, bleu_scores, color=colors)
    plt.xlabel("Experiment Configuration")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score: Comprehensive Ablation Study")
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, score in zip(bars, bleu_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', alpha=0.3)
    
    # Loss 对比
    plt.subplot(1, 2, 2)
    loss_scores = df["loss"].tolist()
    bars = plt.bar(exp_names, loss_scores, color=colors)
    plt.xlabel("Experiment Configuration")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss: Comprehensive Ablation Study")
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, score in zip(bars, loss_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = f"results/ablation_comprehensive_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print("Ablation plot saved to:", plot_path)
    plt.close()
    
    print("\nDetailed Ablation Analysis Results:")
    print("-" * 60)
    baseline_bleu = None
    for idx, row in df.iterrows():
        print(f"Experiment: {row['experiment']}")
        print(f"  Parameters: {row['params']}")
        print(f"  BLEU Score: {row['bleu']:.4f}")
        print(f"  Validation Loss: {row['loss']:.4f}")
        
        if row['experiment'] == 'baseline':
            baseline_bleu = row['bleu']
        elif baseline_bleu is not None:
            diff = row['bleu'] - baseline_bleu
            if diff > 0:
                print(f"  Difference from baseline: +{diff:.4f} (positive: better than baseline)")
            else:
                print(f"  Difference from baseline: {diff:.4f} (negative: worse than baseline)")
        print()

if __name__ == "__main__":
    main()