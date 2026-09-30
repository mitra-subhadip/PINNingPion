import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_model, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    q2_grid = torch.linspace(-5, 5.0, 5000, device=DEVICE)
    with torch.no_grad():
        u, v = model.forward(q2_grid)
        F_sq = u ** 2 + v ** 2

    plt.figure(figsize=(8, 6))
    plt.plot(q2_grid.cpu().numpy(), F_sq.cpu().numpy(), label='PINN Prediction', color='black')

    df_F = pd.read_csv('../benchmark_dataset/Fig1_a.csv')
    plt.plot(df_F['q^2'], df_F['|F(q^2)|^2'], label='1810', color='red')

    plt.xlabel(r'$q^2 \text{ (GeV}^2\text{)}$')
    plt.ylabel(r'$|F_{\pi}(q^2)|^2$')
    plt.title(r'Pion Form Factor $|F_{\pi}(q^2)|^2$ in spacelike, timelike region')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(-1, 2)
    plt.savefig('Benchmark Form Factor.png')
