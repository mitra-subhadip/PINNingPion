"""
Bootstrap error analysis of the rho resonance parameters.

Loops over the bootstrap ensemble and extracts the rho mass and width from
each fit with both methods of resonance_pole.py:

  Method A: 90-degree crossing of the real-axis phase + phase slope,
  Method B: Sheet II pole search (method_B with plot=False, verbose=False,
            and a coarser res=60 grid for speed).

Prints mean +- standard deviation over the ensemble for both methods.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
import resonance_pole as rp
from helper import load_ensemble, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()

    fits = []
    n_skipped = 0

    for model in load_ensemble(DEVICE):
        try:
            m_a, g_a, _ = rp.method_A(model)
            _, m_b, g_b = rp.method_B(model, res=60, plot=False, verbose=False)
        except RuntimeError:  # phase never crossing the required angle
            n_skipped += 1
            continue

        fits.append((m_a, g_a, m_b, g_b))

    m_A, g_A, m_B, g_B = np.array(fits).T
    print(f"\nEnsemble of {len(fits)} bootstrap fits ({n_skipped} skipped)")
    print("Method A (phase crossing):")
    print(f"  m_rho     = {m_A.mean()*1000:.2f} +- {m_A.std()*1000:.2f} MeV")
    print(f"  Gamma_rho = {g_A.mean()*1000:.2f} +- {g_A.std()*1000:.2f} MeV")
    print("Method B (Sheet II pole):")
    print(f"  m_rho     = {m_B.mean()*1000:.2f} +- {m_B.std()*1000:.2f} MeV")
    print(f"  Gamma_rho = {g_B.mean()*1000:.2f} +- {g_B.std()*1000:.2f} MeV")
