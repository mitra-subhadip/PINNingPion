import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from helper import load_model, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    # Post-training Evaluation & Plotting
    df = pd.read_csv("../dataset/spacelike_dataset.csv")
    df_sorted = df.sort_values(by='q2')
    q2_sorted = np.array(df_sorted['q2'])
    fpi_sorted = np.array(df_sorted['Fpi_sq'])

    with torch.no_grad():
        u, v = model.forward(
            torch.tensor(-q2_sorted, dtype=torch.float32).to(DEVICE),
            torch.zeros(len(q2_sorted)).to(DEVICE)
        )
        F_sq = u ** 2 + v ** 2
        q2_spacelike_pred = F_sq.cpu().numpy()

    # Predicted Spacelike |Fpi|^2 vs. Q^2
    plt.figure(figsize=(8, 6))
    plt.plot(-q2_sorted, q2_spacelike_pred, label='PINN Prediction', color='black', zorder=4)

    # Get unique sources to plot them separately
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
    plt.savefig("Spacelike Region.png")
