import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # helper.py and the other scripts live in scripts/
from calculate_differentials import extract_pion_moments
from helper import load_ensemble, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    r_2, r_4, r_6 = np.array([extract_pion_moments(model, DEVICE) for model in load_ensemble(DEVICE)]).T

    print(f"Charge Radius <r^2_pi> = {np.mean(r_2):.6f} +- {np.std(r_2):.6f} fm^2")
    print(f"Curvature     <r^4_pi> = {np.mean(r_4):.6f} +- {np.std(r_4):.6f} fm^4")
    print(f"6th Moment    <r^6_pi> = {np.mean(r_6):.6f} +- {np.std(r_6):.6f} fm^6")
