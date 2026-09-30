import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

from mesonforms.constants import FOUR_M_PI2, CONV_FM2_GEV2
from helper import load_model, set_torch_device


def plot_dispersive_residual(model, device, q2_range=(-2.0, FOUR_M_PI2), num_q=200, num_s=5000):
    """
    Computes the TWICE-SUBTRACTED dispersion relation integral out to infinity 
    to match the exact physics the PINN was trained on.
    """
    model.eval()
    
    R2_PI_FM2 = 0.432
    R2_PI_GEV2 = R2_PI_FM2 * CONV_FM2_GEV2
    
    # INFINITE INTEGRATION GRID (Change of Variables)
    t_vals = np.linspace(1e-5, 1.0, num_s)
    s_vals = FOUR_M_PI2 / t_vals
    
    # Jacobian for ds = (FOUR_M_PI2 / t^2) dt
    ds_dt = FOUR_M_PI2 / (t_vals**2)
    
    # Extract the imaginary part of the form factor
    s_tensor = torch.tensor(s_vals, dtype=torch.float32, device=device)
    with torch.no_grad():
        eps = torch.full_like(s_tensor, 1e-6)
        _, v_pred = model.forward(s_tensor, eps)
        Im_F_s = v_pred.cpu().numpy()
        
    # Evaluation grid for q^2
    q2_vals = np.linspace(q2_range[0], q2_range[1], num_q)
    q2_tensor = torch.tensor(q2_vals, dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred, _ = model.forward(q2_tensor, torch.zeros_like(q2_tensor))
        Re_pinn = u_pred.cpu().numpy()
        
    # Vectorized Twice-Subtracted Dispersion Integral
    Re_disp = np.zeros(num_q)
    
    for i, q2 in enumerate(q2_vals):
        if q2 >= FOUR_M_PI2:
            Re_disp[i] = np.nan
            continue
            
        # TWICE-subtracted integrand: Im[F(s)] / (s^2 * (s - q^2))
        integrand_s = Im_F_s / ((s_vals**2) * (s_vals - q2))
        integrand_t = integrand_s * ds_dt # Jacobian
        integral_val = simpson(integrand_t, x=t_vals)
        
        # TWICE-Subtracted formula
        Re_disp[i] = 1.0 + (1.0 / 6.0) * R2_PI_GEV2 * q2 + (q2**2 / np.pi) * integral_val
        
    # 4. Calculate the residual
    residual = np.abs(Re_pinn - Re_disp)
    
    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Top plot: Re[F] comparison
    ax1.plot(q2_vals, Re_pinn, 'b-', label=r'PINN Direct $\text{Re}[F_\pi(q^2)]$', linewidth=2)
    ax1.plot(q2_vals, Re_disp, 'r--', label=r'Dispersive Integral', linewidth=2)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel(r'$\text{Re}[F_\pi(q^2)]$')
    ax1.set_title('Global Analytic Consistency: PINN Output vs. Dispersion Relation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Residual
    ax2.plot(q2_vals, residual, 'k-', linewidth=1.5)
    ax2.fill_between(q2_vals, 0, residual, color='gray', alpha=0.3)
    ax2.set_xlabel(r'$q^2 \quad [\text{GeV}^2]$') 
    ax2.set_ylabel('Absolute Residual')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.xlim(q2_range[0], q2_range[1])
    plt.savefig('Dispersion.png')

if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)
    plot_dispersive_residual(model, device=DEVICE)
