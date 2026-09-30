import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_model, set_torch_device

plt.rcParams['font.size'] = 14


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    # Spacelike region, excluding q^2 = 0
    q2_grid = torch.linspace(-20.0, 0.0, 2000, device=DEVICE)[:-1]
    with torch.no_grad():
        u, v = model.forward(q2_grid)
        F_mag = torch.sqrt(u ** 2 + v ** 2)

    Q2 = -q2_grid.cpu().numpy()

    plt.figure(figsize=(12, 10))
    plt.plot(Q2, Q2 * F_mag.cpu().numpy(), label='PINN Prediction', color='blue', linewidth=3)

    for curve in 'ABCD':
        df = pd.read_csv(f'../benchmark_dataset/{curve}.csv')
        plt.plot(df['Q^2'], df['Q^2F(Q^2)'], label=f'Curve {curve}', linewidth=3)

    plt.xlabel(r'$Q^2 = -q^2 \text{ (GeV}^2\text{)}$')
    plt.ylabel(r'$Q^2 \cdot |F_{\pi}(Q^2)|$')
    plt.title(r'Asymptotic scaling in Spacelike Region $Q^2 \cdot |F_{\pi}(Q^2)|$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("Benchmark Asymptotic Scaling.png")
