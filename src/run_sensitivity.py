# src/run_sensitivity.py
import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# 定义实验参数空间
EXPERIMENTS = {
    "d_model": [128, 256, 512],
    "num_layers": [2, 4, 6],
    "batch_size": [32, 64, 128]
}

BASE_ARGS = {
    "d_ff": 1024,
    "num_heads": 8,
    "epochs": 5,  # 为加快实验速度，设为5轮
    "lr": 3e-4,
    "device": "cuda"
}

def run_experiment(exp_name, output_dir, **kwargs):
    """运行一次训练实验"""
    cmd = ["python", "-m", "src.train_mt"]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]
        # cmd += [f"--{k.replace('_', '-')}", str(v)]
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
    parser.add_argument("--param", type=str, choices=["d_model", "num_layers", "batch_size"], required=True)
    parser.add_argument("--output_summary", type=str, default="results/sensitivity_summary.csv")
    args = parser.parse_args()

    param_name = args.param
    param_values = EXPERIMENTS[param_name]
    results = []

    print(f"Running sensitivity analysis on: {param_name}")
    for val in param_values:
        exp_name = f"{param_name}_{val}"
        output_dir = f"results/run_experiments/sensitivity/{exp_name}"
        os.makedirs(output_dir, exist_ok=True)

        # 设置参数
        kwargs = BASE_ARGS.copy()
        kwargs[param_name] = val

        metrics = run_experiment(exp_name, output_dir, **kwargs)
        if metrics:
            results.append({"param": param_name, "value": val, "bleu": metrics["bleu"], "loss": metrics["loss"]})
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.output_summary, index=False)
    print("Summary saved to:", args.output_summary)

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["value"], df["bleu"], marker='o')
    plt.xlabel(param_name)
    plt.ylabel("BLEU")
    plt.title(f"BLEU vs {param_name}")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df["value"], df["loss"], marker='x', color='r')
    plt.xlabel(param_name)
    plt.ylabel("Validation Loss")
    plt.title(f"Loss vs {param_name}")
    plt.grid(True)

    plt.tight_layout()
    plot_path = f"results/sensitivity_{param_name}.png"
    plt.savefig(plot_path)
    print("Plot saved to:", plot_path)
    plt.close()

if __name__ == "__main__":
    main()