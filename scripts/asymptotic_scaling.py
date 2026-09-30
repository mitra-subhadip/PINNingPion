import matplotlib.pyplot as plt
import pandas as pd
import torch

from helper import load_model, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)
    
    q2_grid = torch.linspace(-20.0, 0.0, 2000, device=DEVICE)
    with torch.no_grad():
        u, v = model.forward(q2_grid)
        F_mag = torch.sqrt(u ** 2 + v ** 2)
        F_sq = u ** 2 + v ** 2

    df_eval = pd.DataFrame({
        'q2 (GeV^2)': q2_grid.cpu().numpy(),
        'Re(Fpi)': u.cpu().numpy(),
        'Im(Fpi)': v.cpu().numpy(),
        '|Fpi|': F_mag.cpu().numpy(),
        '|Fpi|^2': F_sq.cpu().numpy()
    })

    # Plot 1: Spacelike |Fpi| vs. Q^2
    plt.figure(figsize=(12, 10))
    q2_spacelike_pred = df_eval[df_eval['q2 (GeV^2)'] < 0] 
    plt.plot(-q2_spacelike_pred['q2 (GeV^2)'], -q2_spacelike_pred['q2 (GeV^2)'] * q2_spacelike_pred['|Fpi|'], label='PINN Prediction', color='blue')

    plt.xlabel(r'$Q^2 = -q^2 \text{ (GeV}^2\text{)}$')
    plt.ylabel(r'$Q^2 \cdot |F_{\pi}(Q^2)|$')
    plt.title(r'Asymptotic scaling in Spacelike Region $Q^2 \cdot |F_{\pi}(Q^2)|$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("Asymptotic Scaling.png")
