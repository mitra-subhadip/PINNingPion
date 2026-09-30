import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_model, phase_degrees, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    # q is sqrt(s) in GeV, phase is already in degrees
    df_phase = pd.read_csv('../dataset/colangelo_phase.csv')
    q2 = torch.tensor(df_phase['q'].values**2, dtype=torch.float32, device=DEVICE)

    plt.figure(figsize=(8, 6))
    plt.plot(df_phase['q'], np.abs(phase_degrees(model, q2)), label='PINN')
    plt.plot(df_phase['q'], np.abs(df_phase['phase']), label='Colangelo et.')
    plt.xlabel(r'$q \text{ (GeV)}$')
    plt.ylabel(r'$|arg(F_{\pi}(q^2)|^2)$')
    plt.title(r'$|arg(F_{\pi}(q^2)|^2)$')

    plt.xlim(0.6, 0.825)
    plt.legend()
    plt.savefig('Benchmark Phase.png')
