import matplotlib.pyplot as plt
import numpy as np
import torch

from helper import load_model, set_torch_device
from mesonforms.constants import FOUR_M_PI2

#########################################################
import matplotlib.font_manager as fm
font_path = "../times.ttf"
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.8,
    })
except:
    pass
#########################################################


def plot_complex_form_factor(model, re_range=(-0.2, 1.2), im_range=(-0.4, 0.4), res=150):
    """
    Evaluates the PINN over a grid in the complex q^2 plane and plots 
    both a 3D surface and a heatmap of |F(q^2)| to visualize the branch cut.
    """
    # Create the complex grid
    re_vals = np.linspace(re_range[0], re_range[1], res)
    im_vals = np.linspace(im_range[0], im_range[1], res)
    Re, Im = np.meshgrid(re_vals, im_vals)
    
    # Flatten and convert to tensors
    Re_flat = torch.tensor(Re.flatten(), dtype=torch.float32, device=DEVICE)
    Im_flat = torch.tensor(Im.flatten(), dtype=torch.float32, device=DEVICE)
    
    # Evaluate the model 
    with torch.no_grad():
        u_pred, v_pred = model.forward(Re_flat, Im_flat)
        
    # Calculate magnitude |F(z)| = sqrt(Re(F)^2 + Im(F)^2)
    F_mag_flat = torch.sqrt(u_pred**2 + v_pred**2).cpu().numpy()
    F_mag = F_mag_flat.reshape(res, res)
    
    # Set up the figure for 2 subplots (3D Surface and Heatmap)
    fig = plt.figure(figsize=(16, 6))
    
    # 3D Surface Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # CHANGED: Passed 'viridis' as a string to avoid the cm NameError
    surf = ax1.plot_surface(Re, Im, F_mag, cmap='viridis', 
                            linewidth=0, antialiased=True, alpha=0.85)
    
    # Highlight the branch cut (t >= 4M_pi^2, Im=0)
    cut_re = np.linspace(FOUR_M_PI2, re_range[1], 50)
    cut_im = np.zeros_like(cut_re)
    cut_re_tensor = torch.tensor(cut_re, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_cut, v_cut = model.forward(cut_re_tensor)
        cut_z = torch.sqrt(u_cut**2 + v_cut**2).cpu().numpy()
        
    ax1.plot(cut_re, cut_im, cut_z, color='red', linewidth=2.5, 
             label=r'Branch Cut ($s \geq 4M_\pi^2$)',zorder=200)
    
    ax1.set_xlabel(r'$\text{Re}(s)$ in GeV$^2$', fontsize=14, labelpad=10)
    ax1.set_ylabel(r'$\text{Im}(s)$ in GeV$^2$', fontsize=14, labelpad=10)
    #ax1.set_zlabel(r'$|F_\pi(s)|$', fontsize=14, labelpad=10)
    #ax1.set_title(r'3D Surface of $|F_\pi(q^2)|$')
    ax1.legend(fontsize=14)
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    cbar.ax.set_title(r'$|F_\pi(s)|$', pad=15,x=-0.8)

    # 2D Heatmap
    ax2 = fig.add_subplot(1, 2, 2)
    c = ax2.pcolormesh(Re, Im, F_mag, cmap='viridis', shading='auto', edgecolors='none', rasterized=True)

    # Draw the branch cut line
    ax2.plot([FOUR_M_PI2, re_range[1]], [0, 0], color='red', linewidth=2.5,  label=r'Branch Cut ($s \geq 4M_\pi^2$)')

    # Mark the physical threshold
    ax2.axvline(FOUR_M_PI2, color='white', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel(r'$\text{Re}(s)$ in GeV$^2$', fontsize=14, labelpad=10)
    ax2.set_ylabel(r'$\text{Im}(s)$ in GeV$^2$', fontsize=14, labelpad=10)
    #ax2.set_title(r'Heatmap of $|F_\pi(q^2)|$')
    
    ax2.legend(fontsize=14)
    fig.colorbar(c, ax=ax2)
    
    plt.tight_layout()
    # plt.savefig("Complex_Plane.pdf", dpi=600, bbox_inches="tight")
    plt.savefig("Complex Plane.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    plot_complex_form_factor(model)
