import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_ensemble, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()

    df = pd.read_csv("../dataset/spacelike_dataset.csv")
    q2_sorted = np.array(df.sort_values(by='q2')['q2'])

    # Get unique sources to plot them separately
    plt.figure(figsize=(8, 6))
    sources = df['Source'].unique()

    # Define a list of markers to distinguish sources clearly
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    # Loop through each source and plot its data with error bars
    for i, source in enumerate(sources):
        subset = df[df['Source'] == source]

        plt.errorbar(-subset['q2'], subset['Fpi_sq'], yerr=subset['error'],
                fmt=markers[i % len(markers)], label=source,
                capsize=3, alpha=0.8, linestyle='none', markersize=6)

    plt.xlabel(r'$q^2 \text{ (GeV}^2\text{)}$')
    plt.ylabel(r'$|F_{\pi}(q^2)|^2$')
    plt.title(r'Pion Form Factor $|F_{\pi}(q^2)|^2$ in Spacelike Region')
    plt.legend()
    plt.xlim(-3,0)
    plt.grid(True, linestyle='--', alpha=0.6)

    # PINN Prediction
    q2_plot = torch.tensor(-q2_sorted, dtype=torch.float32, device=DEVICE)

    F = []
    for model in load_ensemble(DEVICE):
        with torch.no_grad():
            u, v = model.forward(q2_plot, torch.zeros_like(q2_plot))
            F.append((u ** 2 + v ** 2).cpu().numpy())

    F_mean = np.mean(F, axis=0)
    F_std = np.std(F, axis=0)

    # Predicted Spacelike |Fpi|^2 vs. Q^2
    plt.plot(-q2_sorted, F_mean, label='PINN Prediction', color='black', zorder=4)
    plt.fill_between(-q2_sorted, F_mean - F_std, F_mean + F_std,
                     color='red', alpha=0.2, label='PINN Uncertainty (±1σ)', zorder=1)
    plt.savefig("Ensemble Spacelike Region.png")
