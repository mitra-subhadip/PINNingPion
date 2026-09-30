import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_ensemble, phase_degrees, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    q2_grid = torch.linspace(0.08, 10.0, 5000, device=DEVICE)
    x_axis = q2_grid.cpu().numpy()

    Phase = [phase_degrees(model, q2_grid) for model in load_ensemble(DEVICE)]
    Phase_mean = np.mean(Phase, axis=0)
    Phase_std = np.std(Phase, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, Phase_mean, label=r'Phase in degrees', color='black', linewidth=2)
    plt.fill_between(x_axis, Phase_mean - Phase_std, Phase_mean + Phase_std,
                     color='red', alpha=0.2, label='PINN Uncertainty (±1σ)', zorder=1)
    plt.xlim(0.08, 2)
    plt.xlabel(r'$q^2 \text{ (GeV}^2\text{)}$')
    plt.ylabel(r'$|arg(F_{\pi}(q^2)|^2)$')
    plt.title(r'$|arg(F_{\pi}(q^2)|^2)$')
    plt.savefig('Ensemble Phase.png')
