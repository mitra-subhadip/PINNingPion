from glob import glob

import numpy as np
import torch
from mesonforms.model import PINN_Fpi


def set_torch_device():
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = 'cpu'

    return DEVICE


def load_model(device, pt_file: str="../weights/final.pt"):
    model = PINN_Fpi().to(device)
    state_dict = torch.load(pt_file, map_location=torch.device(device))
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_ensemble(device, pattern: str="../ensemble/error-no-ablation-*.pt"):
    """Yields every bootstrap fit written by batch.sh."""
    pt_files = sorted(glob(pattern))
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint matches {pattern}; run batch.sh from the repository root first.")

    for pt_file in pt_files:
        yield load_model(device, pt_file)


def phase_degrees(model, q2):
    """Unwrapped phase of the isovector form factor in degrees, for real q2 (upper lip of the cut)."""
    with torch.no_grad():
        u, v = model.forward(q2)
    return np.degrees(np.unwrap(np.arctan2(v.cpu().numpy(), u.cpu().numpy())))
