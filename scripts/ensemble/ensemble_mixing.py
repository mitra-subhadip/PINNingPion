import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from helper import load_ensemble, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    alpha, omega = np.array([(model.alpha.item(), model.phi_omega.item()) for model in load_ensemble(DEVICE)]).T

    print(f"Alpha: {np.mean(alpha)} +- {np.std(alpha)}")
    print(f"Phi_omega: {np.mean(omega)} +- {np.std(omega)}")
