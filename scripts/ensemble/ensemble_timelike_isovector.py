import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_ensemble, set_torch_device
from mesonforms.constants import FOUR_M_PI2


if __name__ == "__main__":
    DEVICE = set_torch_device()

    # Read and filter the timelike dataset
    df = pd.read_csv('../dataset/timelike_dataset.csv')
    df = df[df['I'] == 1]

    sources = df['Source'].unique()
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    # Generate a list of distinct colors using the 'tab10' colormap
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(sources))]

    # Loop through each source and plot its data with error bars
    for i, source in enumerate(sources):
        subset = df[df['Source'] == source]
        plt.errorbar(subset['q2'], subset['Fpi_sq'], yerr=subset['error'],
                    fmt=markers[i % len(markers)], label=source, color=colors[i],
                    capsize=3, alpha=0.9, linestyle='none', markersize=5)

    # PINN Prediction
    q2_plot = torch.linspace(FOUR_M_PI2, 3.2, 500, device=DEVICE)

    F = []
    for model in load_ensemble(DEVICE):
        with torch.no_grad():
            u_I1, v_I1 = model.forward_mixed(q2_plot, torch.zeros_like(q2_plot))
            F.append((u_I1**2 + v_I1**2).cpu().numpy())

    # Plot the prediction
    q2_plot_np = q2_plot.cpu().numpy()
    F_mean = np.mean(F, axis=0)
    F_std = np.std(F, axis=0)

    plt.plot(q2_plot_np, F_mean, label=r'PINN Pure Isovector ($\tau$ limit)', color='black', linewidth=2)
    plt.fill_between(q2_plot_np, F_mean - F_std, F_mean + F_std,
                     color='red', alpha=0.2, label='PINN Uncertainty (±1σ)', zorder=1)

    plt.xlabel(r'$q^2 \text{ (GeV}^2\text{)}$', fontsize=12)
    plt.ylabel(r'$|F_{\pi}(q^2)|^2$', fontsize=12)
    plt.title(r'Pion Form Factor $|F_{\pi}(q^2)|^2$ (pure isovector)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('Ensemble Timelike Isovector.png')
