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
    df_phase = pd.read_csv('../dataset/ananth_phase.csv')
    q = np.array(df_phase['q'])
    phase_degree = phase_degrees(model, torch.tensor(q*q, dtype=torch.float32, device=DEVICE))

    plt.figure(figsize=(8, 6))
    plt.plot(q, np.abs(phase_degree - df_phase['phase']), label='PINN')
    plt.legend()

    plt.ylim(0, 5)

    plt.xlabel(r'$q \text{ (GeV)}$')
    plt.ylabel(r'$|arg(F_{\pi}(q^2)|$')
    plt.title(r'$|arg(F_{\pi}(q^2)|$')

    plt.savefig('Benchmark Roy.png')
    np.savetxt(
        'benchmark_roy.csv',
        np.column_stack([q, phase_degree, df_phase['phase']]),
        delimiter=',',
        header='q (GeV),Phase_deg,Roy',
        comments='',          # otherwise numpy prefixes the header with '# '
    )
