import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from helper import load_model, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    q2_grid = torch.linspace(-5, 5.0, 5000, device=DEVICE)
    with torch.no_grad():
        u, v = model.forward_mixed(q2_grid, q2_grid>0)
        F_sq = u ** 2 + v ** 2
        F_mag = torch.sqrt(F_sq)

    df_eval = pd.DataFrame({
        'q2 (GeV^2)': q2_grid.cpu().numpy(),
        'Re(Fpi)': u.cpu().numpy(),
        'Im(Fpi)': v.cpu().numpy(),
        '|Fpi|': F_mag.cpu().numpy(),
        '|Fpi|^2': F_sq.cpu().numpy()
    })

    plt.figure(figsize=(8, 6))
    plt.plot(df_eval['q2 (GeV^2)'], df_eval['|Fpi|^2'], label='PINN Prediction', color='black', zorder=10)

    # Space like dataset plotting
    df = pd.read_csv("../dataset/spacelike_dataset.csv")
    df_sorted = df.sort_values(by='q2')
    q2_sorted = np.array(df_sorted['q2'])
    fpi_sorted = np.array(df_sorted['Fpi_sq'])

    sources = df['Source'].unique()
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    for i, source in enumerate(sources):
        subset = df[df['Source'] == source]

        plt.errorbar(-subset['q2'], subset['Fpi_sq'], yerr=subset['error'], 
                 fmt=markers[i % len(markers)], label=source, 
                 capsize=3, alpha=0.8, linestyle='none', markersize=6)

    # Timelike dataset plotting
    df_time = pd.read_csv('../dataset/timelike_dataset.csv')
    df_time = df_time[df_time['I'] == 0]
    sources_time = df_time['Source'].unique()
    markers_time = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    cmap_time = plt.get_cmap('tab10') # Distinct colormap for timelike
    
    for i, source in enumerate(sources_time):
        subset = df_time[df_time['Source'] == source]
        plt.errorbar(subset['q2'], subset['Fpi_sq'], yerr=subset['error'], 
                     fmt=markers_time[i % len(markers_time)], label=f"{source} (Timelike)", color=cmap_time(i % 10),
                     capsize=3, alpha=0.9, linestyle='none', markersize=5)

    plt.xlabel(r'$q^2 \text{ (GeV}^2\text{)}$')
    plt.ylabel(r'$|F_{\pi}(q^2)|$')
    plt.title(r'Pion Form Factor $|F_{\pi}(q^2)|$ in spacelike, timelike region')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(-1, 1.5)
    plt.savefig('Continuous region - interference.png')
