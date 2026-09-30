import matplotlib.pyplot as plt
import pandas as pd
import torch

from helper import load_model, set_torch_device
from mesonforms.constants import FOUR_M_PI2


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    q2_plot = torch.linspace(FOUR_M_PI2, 3.2, 5000, device=DEVICE)

    u_I1, v_I1 = model.forward_mixed(q2_plot, torch.zeros_like(q2_plot))
    Fpi_sq_I1 = (u_I1**2 + v_I1**2).detach().cpu().numpy()

    u_mixed, v_mixed = model.forward_mixed(q2_plot, torch.ones_like(q2_plot))
    Fpi_sq_mixed = (u_mixed**2 + v_mixed**2).detach().cpu().numpy()

    # Predicted values
    plt.figure(figsize=(8, 6))
    q2_plot_np = q2_plot.cpu().numpy()
    plt.plot(q2_plot_np, Fpi_sq_I1, label=r'PINN Pure Isovector ($\tau$ limit)', color='green', linewidth=2)

    # Read and filter the timelike dataset
    df = pd.read_csv('../dataset/uncorrected_timelike_dataset.csv')
    df = df[df['I'] == 1]
    
    sources = df['Source'].unique()
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    # Generate a list of distinct colors using the 'tab10' colormap
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(sources))]

    # Loop through each source and plot its data with error bars
    for i, source in enumerate(sources):
        subset = df[df['Source'] == source]
        plt.errorbar(subset['q2'], subset['Fpi_sq'], yerr=subset['error'], 
                    fmt=markers[i % len(markers)], label=source, color=colors[i],
                    capsize=3, alpha=0.9, linestyle='none', markersize=5)

    plt.xlabel(r'$q^2 \text{ (GeV}^2\text{)}$', fontsize=12)
    plt.ylabel(r'$|F_{\pi}(q^2)|^2$', fontsize=12)
    plt.title(r'Pion Form Factor $|F_{\pi}(q^2)|^2$ (pure isovector)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.xlim(0.3, 5) 
    # plt.yscale('log')

    plt.tight_layout()
    plt.savefig('Timelike Isovector.png')
