import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_model, phase_degrees, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()

    q2_grid = torch.linspace(0.08, 2.0, 5000, device=DEVICE)
    q = torch.sqrt(q2_grid).cpu().numpy()

    plt.figure(figsize=(8, 6))
    for pt_file, label in [
        ('../weights/ablation-watson.pt', r'PINN ($L_{\text{Watson}=0}$)'),
        ('../weights/final.pt', r'PINN'),
    ]:
        model = load_model(DEVICE, pt_file=pt_file)
        plt.plot(q, phase_degrees(model, q2_grid), label=label)

    df_phase = pd.read_csv('../dataset/colangelo_phase.csv')
    plt.plot(df_phase['q'], df_phase['phase'], label='Roy', linestyle='dashed')
    plt.legend()

    plt.xlim(0.36, 1.1)
    plt.xlabel(r'$q \text{ (GeV}\text{)}$')
    plt.ylabel(r'$|arg(F_{\pi}(q^2)|^2)$')
    plt.title(r'$|arg(F_{\pi}(q^2)|^2)$')
    plt.savefig('Benchmark Watson Ablation.png')
