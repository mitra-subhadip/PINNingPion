import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.colors import LogNorm

from helper import load_model, set_torch_device

def hunt_for_zeros(model, re_range=(-20.0, 20.0), im_range=(-20.0, 20.0), res=1000, zero_tol=1e-4):
    """
    Scans the complex plane for roots where |F_pi(z)| = 0.
    Uses a coarse grid search followed by Scipy optimization for refinement.
    """
    model.eval()
    
    # Coarse Grid Search
    re_vals = np.linspace(re_range[0], re_range[1], res)
    im_vals = np.linspace(im_range[0], im_range[1], res)
    Re, Im = np.meshgrid(re_vals, im_vals)
    
    Re_flat = torch.tensor(Re.flatten(), dtype=torch.float32, device=DEVICE)
    Im_flat = torch.tensor(Im.flatten(), dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        u_pred, v_pred = model.forward(Re_flat, Im_flat)
        
    F_mag_sq = (u_pred**2 + v_pred**2).cpu().numpy().reshape(res, res)
    F_mag = np.sqrt(F_mag_sq)
    
    # Find the global minimum on the grid
    min_idx = np.unravel_index(np.argmin(F_mag), F_mag.shape)
    min_re_grid = re_vals[min_idx[1]]
    min_im_grid = im_vals[min_idx[0]]
    min_val_grid = F_mag[min_idx]
    
    print(f"Grid Minimum Found: |F| = {min_val_grid:.6e} at z = {min_re_grid:.4f} + {min_im_grid:.4f}i")
    
    # Fine-tune with Scipy Optimizer
    def objective(z):
        re_t = torch.tensor([z[0]], dtype=torch.float32, device=DEVICE)
        im_t = torch.tensor([z[1]], dtype=torch.float32, device=DEVICE)
        u, v = model.forward(re_t, im_t)
        return (u**2 + v**2).item()
    
    # Start optimization from the grid minimum
    res_opt = minimize(objective, x0=[min_re_grid, min_im_grid], method='Nelder-Mead')
    
    opt_re, opt_im = res_opt.x
    opt_val = np.sqrt(res_opt.fun)
    
    print(f"Optimized Minimum:  |F| = {opt_val:.6e} at z = {opt_re:.4f} + {opt_im:.4f}i")
    
    # Determine if it's a true zero or just a minimum. A minimum reached outside
    # the scanned window is the asymptotic fall-off F -> 0, not a zero of this domain.
    inside = re_range[0] <= opt_re <= re_range[1] and im_range[0] <= opt_im <= im_range[1]
    found_zero = inside and opt_val < zero_tol
    if found_zero:
        print(f"\n>>> ALERT: Possible zero detected at z = {opt_re:.4f} + {opt_im:.4f}i <<<")
    else:
        print("\n>>> No zeros found within tolerance. The Zero Hypothesis holds in this domain. <<<")
        
    # Plotting the Log-Magnitude to visualize the landscape
    plt.figure(figsize=(10, 8))
    # Using LogNorm to make zeros (sinkholes) visually obvious
    c = plt.pcolormesh(Re, Im, F_mag, norm=LogNorm(vmin=max(1e-5, F_mag.min()), vmax=F_mag.max()), cmap='magma', shading='auto')
    
    # Only draw the marker and legend if a zero actually exists
    if found_zero:
        plt.plot(opt_re, opt_im, 'w*', markersize=15, markeredgecolor='k', label='Detected Zero')
        plt.legend(loc='upper right')
        
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(c, label=r'$|F_\pi(q^2)|$ (Log Scale)')
    plt.xlabel(r'$\text{Re}(q^2)$ [GeV$^2$]')
    plt.ylabel(r'$\text{Im}(q^2)$ [GeV$^2$]')
    plt.title(r'Log-Magnitude Landscape of $F_\pi(q^2)$')
    
    # Set the limits to the actual computed grid range to prevent distortion
    plt.xlim(re_range)
    plt.ylim(im_range)
    
    plt.tight_layout()
    plt.savefig("No Zeros Complex.png", dpi=300)
    
    return found_zero, (opt_re, opt_im)

if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)
    has_zero, zero_loc = hunt_for_zeros(model)