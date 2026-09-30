import torch
import numpy as np
from scipy.integrate import simpson
from scipy.special import spence

from mesonforms.constants import FOUR_M_PI2
from helper import load_model, set_torch_device

# ==========================================
# FSR KINEMATIC HELPER FUNCTIONS
# ==========================================
def dilog(x):
    return spence(1 - x)

def F(x):
    term1 = -4 * dilog(x)
    term2 = 4 * dilog(-x)
    term3 = 2 * np.log(x) * np.log((1 + x) / (1 - x))
    term4 = 3 * dilog((1 + x) / 2)
    term5 = -3 * dilog((1 - x) / 2)
    term6 = (np.pi**2) / 2
    return term1 + term2 + term3 + term4 + term5 + term6

def eta(s, m_pi):
    sigma = np.sqrt(1.0 - (4.0 * m_pi**2) / s)
    sigma2 = sigma**2
    sigma3 = sigma**3
    
    term1 = 3 * (1 + sigma2) / (2 * sigma2)
    term2 = -4 * np.log(sigma)
    term3 = 6 * np.log((1 + sigma) / 2)
    term4 = ((1 + sigma2) / sigma) * F(sigma)
    
    numerator = (1 - sigma) * (3 + 3*sigma - 7*sigma2 + 5*sigma3)
    denominator = 4 * sigma3
    term5 = - (numerator / denominator) * np.log((1 + sigma) / (1 - sigma))
    
    return term1 + term2 + term3 + term4 + term5

def calculate_amu_hvp_pinn_fsr(model, device, s_max, num_s=10000):
    """
    Calculates the two-pion HVP contribution to the muon anomalous magnetic moment
    using the learned form factor from the PINN model, including FSR corrections.
    """
    model.eval()
    
    # Physics constants (in GeV)
    M_MU = 0.10565837
    M_PI = np.sqrt(FOUR_M_PI2 / 4.0) 
    ALPHA = 1.0 / 137.035999
    
    # FINITE INTEGRATION GRID
    # Start microscopically above the 4m_pi^2 threshold to avoid log(0) in eta(s)
    s_start = FOUR_M_PI2 + 1e-7
    s_vals = np.linspace(s_start, s_max, num_s)
    
    # Extract PINN Form Factor |F_pi(s)|^2 along the cut
    s_tensor = torch.tensor(s_vals, dtype=torch.float32, device=device)
    with torch.no_grad():
        eps = torch.full_like(s_tensor, 1e-6)
        u_pred, v_pred = model.forward(s_tensor, eps)
        
        u_np = u_pred.cpu().numpy().squeeze()
        v_np = v_pred.cpu().numpy().squeeze()
        F_pi_sq = u_np**2 + v_np**2
        
    # Calculate Kinematic Functions
    sigma_mu = np.sqrt(1.0 - 4.0 * M_MU**2 / s_vals)
    x_s = (1.0 - sigma_mu) / (1.0 + sigma_mu)
    x_s = np.clip(x_s, 1e-12, 1.0)
    
    # K_hat(s) 
    term1 = (x_s**2 / 2.0) * (2.0 - x_s**2)
    term2 = ((1.0 + x_s**2) * (1.0 + x_s)**2 / x_s**2) * (np.log(1.0 + x_s) - x_s + x_s**2 / 2.0)
    term3 = ((1.0 + x_s) / (1.0 - x_s)) * (x_s**2) * np.log(x_s)
    K_hat = (3.0 * s_vals / M_MU**2) * (term1 + term2 + term3)
    
    # sigma_pi(s)
    sigma_pi_vals = np.sqrt(1.0 - FOUR_M_PI2 / s_vals)
    
    # R_had(s) unperturbed 
    R_had_bare = 0.25 * (sigma_pi_vals**3) * F_pi_sq
    
    # APPLY FSR CORRECTION
    eta_vals = eta(s_vals, M_PI)
    fsr_factor = 1.0 + (ALPHA / np.pi) * eta_vals
    R_had_fsr = R_had_bare * fsr_factor
    
    # Integrate using Simpson's Rule directly over s
    integrand_s = (K_hat / (s_vals**2)) * R_had_fsr
    
    # Integrate from s_start to s_max
    integral_val = simpson(integrand_s, x=s_vals)
    
    # Final a_mu Calculation with prefactor: (alpha * m_mu / 3*pi)^2
    prefactor = (ALPHA * M_MU / (3.0 * np.pi))**2
    return prefactor * integral_val


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    for s_max in (0.3969, 1, 100):
        a_mu_pipi = calculate_amu_hvp_pinn_fsr(model, DEVICE, s_max)

        # Standard reporting unit is 10^-10
        print(f"Muon Anomalous Magnetic Moment (2π HVP + FSR) up to s = {s_max} GeV^2")
        print(f"a_μ^(ππ) = {a_mu_pipi * 1e10:.2f} x 10^-10")
