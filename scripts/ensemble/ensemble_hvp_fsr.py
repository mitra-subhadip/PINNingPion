import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_ensemble, set_torch_device
from hvp_fsr import calculate_amu_hvp_pinn_fsr


if __name__ == "__main__":
    DEVICE = set_torch_device()
    S_MAX = 0.3969

    a_mu = np.array([calculate_amu_hvp_pinn_fsr(model, DEVICE, S_MAX) for model in load_ensemble(DEVICE)])
    print(f"a_mu: {np.mean(a_mu)} +- {np.std(a_mu)}")
