import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.colors import LogNorm

from helper import load_model, set_torch_device
from mesonforms.constants import FOUR_M_PI2, M_PI

W_RHO = 0.790353 - 0.728033j            # rho pole image in w (sheet II, |w| > 1)
A = [-3.527647, +4.755645, -2.937709, +0.707660]
B = [-1.554048, +0.758975]

FIT_RE = (-1.5, 2.5)                    # complex-grid window used during the fit
FIT_IM = (-1.0, 1.0)


def F_pade_I1(s_complex):
    """Pure isovector (I=1) parametrization, complex s on the first sheet."""
    r = np.sqrt(FOUR_M_PI2 - s_complex)
    w = (r - 2 * M_PI) / (r + 2 * M_PI)
    num = 1 + A[0] * w + A[1] * w ** 2 + A[2] * w ** 3 + A[3] * w ** 4
    den = (1 - w / W_RHO) * (1 - w / np.conj(W_RHO)) * (1 + B[0] * w + B[1] * w ** 2)
    return num / den


def plot_3d_surfaces(model, device, out_file, re_range=(-0.2, 1.2), im_range=(-0.4, 0.4), res=150):
    """3D surfaces of |F|: PINN, Pade, and their absolute difference | |F_PINN| - |F_Pade| |."""
    re_vals = np.linspace(re_range[0], re_range[1], res)
    im_vals = np.linspace(im_range[0], im_range[1], res)
    Re, Im = np.meshgrid(re_vals, im_vals)

    with torch.no_grad():
        u, v = model.forward(torch.tensor(Re.flatten(), dtype=torch.float32, device=device),
                             torch.tensor(Im.flatten(), dtype=torch.float32, device=device))
    F_pinn = np.sqrt(u.cpu().numpy() ** 2 + v.cpu().numpy() ** 2).reshape(Re.shape)

    S = Re + 1j * Im
    S = np.where((np.abs(Im) < 1e-9) & (Re >= FOUR_M_PI2), Re + 1e-7j, S)
    F_pade = np.abs(F_pade_I1(S))

    diff = np.abs(F_pinn - F_pade)

    # Branch cut trace (upper lip) for the two |F| surfaces
    cut_re = np.linspace(FOUR_M_PI2, re_range[1], 50)
    with torch.no_grad():
        u_c, v_c = model.forward(torch.tensor(cut_re, dtype=torch.float32, device=device))
    cut_pinn = np.sqrt(u_c.cpu().numpy() ** 2 + v_c.cpu().numpy() ** 2)
    cut_pade = np.abs(F_pade_I1(cut_re + 1e-7j))

    fig = plt.figure(figsize=(22, 6))
    panels = [(F_pinn, cut_pinn, 'viridis', r'PINN $|F_\pi(s)|$'),
              (F_pade, cut_pade, 'viridis', r'Pade $|F_\pi(s)|$'),
              (diff, None, 'magma', r'$\left|\,|F^{\rm PINN}_\pi| - |F^{\rm Pade}_\pi|\,\right|$')]
    for i, (Z, cut_z, cmap, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        surf = ax.plot_surface(Re, Im, Z, cmap=cmap, linewidth=0,
                               antialiased=True, alpha=0.85)
        if cut_z is not None:
            ax.plot(cut_re, np.zeros_like(cut_re), cut_z, color='red', linewidth=2.5,
                    label=r'Branch Cut ($s \geq 4M_\pi^2$)', zorder=200)
            ax.legend(fontsize=11)
        ax.set_xlabel(r'$\text{Re}(s)$ in GeV$^2$', fontsize=13, labelpad=10)
        ax.set_ylabel(r'$\text{Im}(s)$ in GeV$^2$', fontsize=13, labelpad=10)
        ax.set_title(title, fontsize=14)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"saved {out_file}")


def plot_real_line(model, device, out_file, re_range=(-5.0, 5.0), n_points=4000):
    """Real-axis restriction (Im s -> 0+) of the complex-plane Pade vs the PINN."""
    s = np.linspace(re_range[0], re_range[1], n_points)
    im = np.where(s >= FOUR_M_PI2, 1e-7, 0.0)   # +i*eps on the cut, exactly real elsewhere

    with torch.no_grad():
        u, v = model.forward(torch.tensor(s, dtype=torch.float32, device=device),
                             torch.tensor(im, dtype=torch.float32, device=device))
    F_pinn = np.sqrt(u.cpu().numpy() ** 2 + v.cpu().numpy() ** 2)
    F_pade = np.abs(F_pade_I1(s + 1j * im))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(s, F_pinn, 'k-', lw=2, label='PINN')
    ax.plot(s, F_pade, 'r--', lw=1.5, label=r'Pade (complex-plane formula, Im $s \to 0^+$)')
    ax.axvline(FOUR_M_PI2, color='gray', linestyle=':', alpha=0.8)
    ax.axvspan(FOUR_M_PI2, re_range[1], color='red', alpha=0.06,
               label=r'cut $s \geq 4M_\pi^2$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$s$ [GeV$^2$]', fontsize=13)
    ax.set_ylabel(r'$|F_\pi(s)|$', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pade parametrization vs PINN in the complex s plane.")
    parser.add_argument("--weights", default="../weights/final.pt", help="Path to the model weights")
    parser.add_argument("--range", type=float, default=5.0, help="Half-width R of the window (-R,R)x(-R,R)")
    parser.add_argument("--res", type=int, default=240, help="Grid resolution per axis")
    parser.add_argument("--plot", default="Pade Complex Plane.png", help="Output plot file")
    parser.add_argument("--plot3d", default="Pade Complex Plane 3D.png", help="Output 3D surface plot file")
    parser.add_argument("--plotline", default="Pade Real Line.png", help="Output real-axis plot file")
    args = parser.parse_args()

    DEVICE = set_torch_device()
    model = load_model(DEVICE, pt_file=args.weights)

    R = args.range
    re_vals = np.linspace(-R, R, args.res)
    im_vals = np.linspace(-R, R, args.res)
    Re, Im = np.meshgrid(re_vals, im_vals)

    with torch.no_grad():
        u, v = model.forward(torch.tensor(Re.flatten(), dtype=torch.float32, device=DEVICE),
                             torch.tensor(Im.flatten(), dtype=torch.float32, device=DEVICE))
    F_pinn = (u.cpu().numpy() + 1j * v.cpu().numpy()).reshape(Re.shape)

    # Same +i*eps prescription on the cut as model.forward
    S = Re + 1j * Im
    S = np.where((np.abs(Im) < 1e-9) & (Re >= FOUR_M_PI2), Re + 1e-7j, S)
    F_pade = F_pade_I1(S)

    E = np.abs(F_pade - F_pinn) / np.abs(F_pinn)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    norm = LogNorm(vmin=max(np.abs(F_pinn).min(), 1e-2), vmax=np.abs(F_pinn).max())
    for ax, Z, title in [(axes[0], np.abs(F_pinn), r'$|F_\pi|$  PINN'),
                         (axes[1], np.abs(F_pade), r'$|F_\pi|$  Pade parametrization')]:
        pc = ax.pcolormesh(Re, Im, Z, cmap='viridis', norm=norm, shading='auto')
        plt.colorbar(pc, ax=ax)
        ax.plot([FOUR_M_PI2, R], [0, 0], 'r-', lw=2, label=r'cut $s \geq 4m_\pi^2$')
        ax.set_xlabel(r'Re $s$ [GeV$^2$]')
        ax.set_ylabel(r'Im $s$ [GeV$^2$]')
        ax.set_title(title)
        ax.legend(loc='lower left')

    pc = axes[2].pcolormesh(Re, Im, E * 100, cmap='magma', vmin=0, vmax=25, shading='auto')
    plt.colorbar(pc, ax=axes[2], label='relative error [%]')
    axes[2].add_patch(mpatches.Rectangle((FIT_RE[0], FIT_IM[0]), FIT_RE[1] - FIT_RE[0],
                                         FIT_IM[1] - FIT_IM[0], fill=False, ec='cyan',
                                         lw=1.8, label='fit region'))
    axes[2].set_xlabel(r'Re $s$ [GeV$^2$]')
    axes[2].set_ylabel(r'Im $s$ [GeV$^2$]')
    axes[2].set_title(r'$|F|$ relative error')
    axes[2].legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)

    inside = (Im >= FIT_IM[0]) & (Im <= FIT_IM[1]) & (Re >= FIT_RE[0]) & (Re <= FIT_RE[1])
    print(f"inside fit region : mean={E[inside].mean()*100:.2f}%  max={E[inside].max()*100:.2f}%")
    print(f"outside fit region: mean={E[~inside].mean()*100:.2f}%  max={E[~inside].max()*100:.2f}%")
    print(f"saved {args.plot}")

    plot_3d_surfaces(model, DEVICE, args.plot3d)
    plot_real_line(model, DEVICE, args.plotline, re_range=(-R, R))
