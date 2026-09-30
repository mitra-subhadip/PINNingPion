"""
Compares Arg(F_pi) of the closed-form Pade parametrization against a trained PINN
over the complex s plane, in the style of complex_plane_arg.py.

The phase is NOT a separate fit: for an analytic form factor, |F| and Arg(F) are
tied together (Cauchy-Riemann / dispersion relations, Watson's theorem on the cut),
so the parametrization of the phase is the argument of the same rational formula
(imported from pade_complex_plane.py):

    Arg F(s) = Arg N(w) - Arg(1 - w/w_rho) - Arg(1 - w/conj(w_rho)) - Arg(1 + b1 w + b2 w^2)

evaluated at w = w(s).  On the upper lip of the cut this is the elastic P-wave
phase shift delta_1^1(s) (Watson's theorem).

Outputs a heatmap trio (PINN, Pade, wrap-safe difference) over (-R,R)^2 and a 3D
surface trio in the resonance-region window of complex_plane_arg.py.

Usage:  python pade_complex_plane_arg.py [--weights ../weights/final.pt] [--range 5]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from helper import load_model, set_torch_device
from mesonforms.constants import FOUR_M_PI2
from pade_complex_plane import F_pade_I1


def eval_args(model, device, Re, Im):
    """Arg F from the PINN and from the Pade formula on a meshgrid (wrap-consistent)."""
    with torch.no_grad():
        u, v = model.forward(torch.tensor(Re.flatten(), dtype=torch.float32, device=device),
                             torch.tensor(Im.flatten(), dtype=torch.float32, device=device))
    F_pinn = (u.cpu().numpy() + 1j * v.cpu().numpy()).reshape(Re.shape)

    S = Re + 1j * Im
    S = np.where((np.abs(Im) < 1e-9) & (Re >= FOUR_M_PI2), Re + 1e-7j, S)
    F_pade = F_pade_I1(S)

    # wrap-safe phase difference: argument of the ratio
    dphi = np.angle(F_pade / F_pinn)
    return np.angle(F_pinn), np.angle(F_pade), dphi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pade parametrization vs PINN: Arg(F) in the complex s plane.")
    parser.add_argument("--weights", default="../weights/final.pt", help="Path to the model weights")
    parser.add_argument("--range", type=float, default=5.0, help="Half-width R of the heatmap window")
    parser.add_argument("--res", type=int, default=240, help="Grid resolution per axis")
    parser.add_argument("--plot", default="Pade Complex Plane Phase.png", help="Output heatmap file")
    parser.add_argument("--plot3d", default="Pade Complex Plane Phase 3D.png", help="Output 3D surface file")
    args = parser.parse_args()

    DEVICE = set_torch_device()
    model = load_model(DEVICE, pt_file=args.weights)

    # ---------- heatmap trio over the wide window ----------
    R = args.range
    Re, Im = np.meshgrid(np.linspace(-R, R, args.res), np.linspace(-R, R, args.res))
    arg_pinn, arg_pade, dphi = eval_args(model, DEVICE, Re, Im)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, Z, title in [(axes[0], arg_pinn, r'$\mathrm{Arg}\,F_\pi(s)$  PINN'),
                         (axes[1], arg_pade, r'$\mathrm{Arg}\,F_\pi(s)$  Pade')]:
        c = ax.pcolormesh(Re, Im, Z, cmap='twilight', vmin=-np.pi, vmax=np.pi, shading='auto')
        plt.colorbar(c, ax=ax, label='rad')
        ax.plot([FOUR_M_PI2, R], [0, 0], color='red', linewidth=2, label=r'Branch Cut ($s \geq 4M_\pi^2$)')
        ax.set_xlabel(r'$\mathrm{Re}(s)$ in GeV$^2$')
        ax.set_ylabel(r'$\mathrm{Im}(s)$ in GeV$^2$')
        ax.set_title(title)
        ax.legend(loc='lower left', fontsize=9)
    c = axes[2].pcolormesh(Re, Im, np.degrees(np.abs(dphi)), cmap='magma', vmin=0, vmax=5, shading='auto')
    plt.colorbar(c, ax=axes[2], label=r'$|\Delta\,\mathrm{Arg}\,F|$ [deg]')
    axes[2].set_xlabel(r'$\mathrm{Re}(s)$ in GeV$^2$')
    axes[2].set_ylabel(r'$\mathrm{Im}(s)$ in GeV$^2$')
    axes[2].set_title('phase difference (wrap-safe)')
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)
    print(f"saved {args.plot}")

    # ---------- 3D surfaces in the resonance-region window of complex_plane_arg.py ----------
    Re3, Im3 = np.meshgrid(np.linspace(-0.2, 1.2, 150), np.linspace(-0.4, 0.4, 150))
    a_pinn, a_pade, dphi3 = eval_args(model, DEVICE, Re3, Im3)

    cut_re = np.linspace(FOUR_M_PI2, 1.2, 50)
    with torch.no_grad():
        u_c, v_c = model.forward(torch.tensor(cut_re, dtype=torch.float32, device=DEVICE))
    cut_pinn = np.arctan2(v_c.cpu().numpy(), u_c.cpu().numpy())
    cut_pade = np.angle(F_pade_I1(cut_re + 1e-7j))

    fig = plt.figure(figsize=(22, 6))
    panels = [(a_pinn, cut_pinn, 'twilight', r'PINN $\mathrm{Arg}\,F_\pi(s)$'),
              (a_pade, cut_pade, 'twilight', r'Pade $\mathrm{Arg}\,F_\pi(s)$'),
              (np.degrees(np.abs(dphi3)), None, 'magma', r'$|\Delta\,\mathrm{Arg}\,F_\pi|$ [deg]')]
    for i, (Z, cut_z, cmap, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        surf = ax.plot_surface(Re3, Im3, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.85)
        if cut_z is not None:
            ax.plot(cut_re, np.zeros_like(cut_re), cut_z, color='red', linewidth=2.5,
                    label=r'Branch Cut ($s \geq 4M_\pi^2$)', zorder=200)
            ax.legend(fontsize=11)
        ax.set_xlabel(r'$\mathrm{Re}(s)$ in GeV$^2$', fontsize=13, labelpad=10)
        ax.set_ylabel(r'$\mathrm{Im}(s)$ in GeV$^2$', fontsize=13, labelpad=10)
        ax.set_title(title, fontsize=14)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(args.plot3d, dpi=300, bbox_inches="tight")
    print(f"saved {args.plot3d}")

    # ---------- summary statistics ----------
    print(f"phase error over ({-R},{R})^2 window: mean={np.degrees(np.abs(dphi)).mean():.2f} deg, "
          f"p95={np.percentile(np.degrees(np.abs(dphi)), 95):.2f} deg, max={np.degrees(np.abs(dphi)).max():.2f} deg")
    dcut = np.degrees(np.abs(np.angle(F_pade_I1(cut_re + 1e-7j) /
                                      (u_c.cpu().numpy() + 1j * v_c.cpu().numpy()))))
    print(f"Watson phase delta_1^1 on the cut (up to 1.2 GeV^2): mean={dcut.mean():.2f} deg, max={dcut.max():.2f} deg")
