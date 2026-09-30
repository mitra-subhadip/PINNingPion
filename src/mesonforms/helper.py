import math
import numpy as np
import scipy
import torch
from .constants import FOUR_M_PI2


def get_experimental_phase_interpolator(q2_points, deg_points):
    """
    Returns a function that interpolates the precise P-wave pi-pi phase shift delta_1^1.
    Data points are taken from the Garcia-Martin et al. (GKPY) / Bern analysis 
    standard dispersive fits, which are more accurate than a simple Breit-Wigner.
    """
    if q2_points[0] > FOUR_M_PI2:
        # Inject the theoretical (t_th, 0.0) point to anchor the cubic spline
        q2_points = np.insert(q2_points, 0, FOUR_M_PI2)
        deg_points = np.insert(deg_points, 0, 0.0)
        
        # Optional but highly recommended: Inject a small "helper" point slightly above 
        # threshold to enforce the flat P-wave (p^3) purely kinematic rise.
        # This prevents the cubic spline from oscillating below zero.
        q2_points = np.insert(q2_points, 1, FOUR_M_PI2 + 0.01)
        deg_points = np.insert(deg_points, 1, 0.01)

    rad_points = np.deg2rad(deg_points)
    
    # Create a linear interpolator (safe for gradients if wrapped carefully, 
    # but usually we pre-calculate this before the training loop or assume fixed targets)
    # fill_value="extrapolate" handles high s, assuming asymptotic approach to 180 (pi)
    return scipy.interpolate.interp1d(q2_points, rad_points, kind='linear', fill_value="extrapolate")


def experimental_delta11(q2_tensor, phase_interpolator, device):
    """
    Computes the experimental phase shift for a tensor of q^2 values.
    Input: q2_tensor (Torch Tensor)
    Output: delta_11 (Torch Tensor) in radians
    """
    # Detach to convert to numpy for scipy interpolation
    # The phase target is a fixed physical truth, so we don't need gradients *through* the lookup.
    q2_np = q2_tensor.detach().cpu().numpy()
    
    # Handle the cut: Phase is 0 below 4*m_pi^2
    mask = q2_np > FOUR_M_PI2
    phase_np = np.zeros_like(q2_np)
    
    if np.any(mask):
        phase_np[mask] = phase_interpolator(q2_np[mask])
    
    return torch.tensor(phase_np, dtype=torch.float32, device=device)


def strong_coupling_2loop(Q, Lambda=0.226, n_f=5):
    """
    Calculates the 2-loop alpha_s at energy scale Q.
    Lambda ~ 0.226 GeV is tuned to hit alpha_s(Mz) ~ 0.118.
    """
    # Beta function coefficients
    beta_0 = (33 - 2 * n_f) / (12 * math.pi)  # This is actually beta_0/4*pi
    beta_1 = (153 - 19 * n_f) / (24 * math.pi**2)
    
    # Logarithmic term
    L = torch.log(Q**2 / Lambda**2)
    
    # 2-loop expansion
    alpha_s = (1 / (beta_0 * L)) * (1 - (beta_1 / beta_0**2) * torch.log(L) / L)
    return alpha_s
