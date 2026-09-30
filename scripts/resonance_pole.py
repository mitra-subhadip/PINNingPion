import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from scipy.optimize import minimize

from helper import load_model, set_torch_device
from mesonforms.constants import FOUR_M_PI2


def evaluate_F1(model, s_re, s_im):
    """Evaluates the Sheet I form factor F_1(s) at complex s = s_re + i*s_im."""
    device = model.alpha.device
    re_t = torch.as_tensor(np.atleast_1d(s_re), dtype=torch.float32, device=device)
    im_t = torch.as_tensor(np.atleast_1d(s_im), dtype=torch.float32, device=device)
    with torch.no_grad():
        u, v = model.forward(re_t, im_t)
    return u.cpu().numpy() + 1j * v.cpu().numpy()


def real_axis_phase(model, s_min=FOUR_M_PI2 + 1e-4, s_max=1.2, n_points=20000):
    """Unwrapped isovector phase delta_1^1(s) = arg F_1(s + i*eps) on the cut (Watson)."""
    s_grid = np.linspace(s_min, s_max, n_points)
    F1 = evaluate_F1(model, s_grid, np.full_like(s_grid, 1e-7))
    delta = np.unwrap(np.arctan2(F1.imag, F1.real))
    return s_grid, delta


def crossing(s_grid, delta, target):
    """First s where the phase crosses `target` (linear interpolation)."""
    above = np.where(delta >= target)[0]
    if len(above) == 0 or above[0] == 0:
        raise RuntimeError(f"Phase never crosses {np.degrees(target):.0f} degrees on the scanned interval.")
    i = above[0]
    frac = (target - delta[i - 1]) / (delta[i] - delta[i - 1])
    return s_grid[i - 1] + frac * (s_grid[i] - s_grid[i - 1])


def method_A(model):
    """
    Coarse extraction from the real-axis phase shift.
    m_rho: sqrt(s) where the phase crosses 90 degrees.
    Gamma_rho: from the phase slope, since tan(delta) = m*Gamma / (m^2 - s)
    implies d(delta)/ds = 1 / (m*Gamma) at s = m^2.
    Cross-check: sqrt(s) spread between the 45 and 135 degree crossings.
    """
    s_grid, delta = real_axis_phase(model)

    s_res = crossing(s_grid, delta, np.pi / 2)
    m_rho = np.sqrt(s_res)

    ddelta_ds = np.gradient(delta, s_grid)
    slope = np.interp(s_res, s_grid, ddelta_ds)
    gamma_rho = 1.0 / (m_rho * slope)

    gamma_spread = np.sqrt(crossing(s_grid, delta, 3 * np.pi / 4)) - np.sqrt(crossing(s_grid, delta, np.pi / 4))
    return m_rho, gamma_rho, gamma_spread


def fit_cot_delta(model, delta_window=(np.radians(30), np.radians(150)), order=3):
    """
    Fits cot(delta(s)) with a polynomial C(s) over the resonance region.
    cot(delta) is smooth through the resonance (it crosses zero linearly at
    s = m_rho^2), so a low-order polynomial gives a controlled analytic
    continuation of the S-matrix, S(s) = (C(s) + i) / (C(s) - i) = e^{2i*delta}.
    """
    s_grid, delta = real_axis_phase(model)
    mask = (delta > delta_window[0]) & (delta < delta_window[1])
    C = np.poly1d(np.polyfit(s_grid[mask], 1.0 / np.tan(delta[mask]), order))
    return C, s_grid[mask], delta[mask]


def sheet_II_inverse(model, s_re, s_im, C):
    """
    |1 / F_2(s)|^2 on Sheet II. Crossing the elastic cut downwards gives
    F_2(s) = S(s) * F_1(s), with S(s) = e^{2i*delta(s)} continued analytically
    via the cot(delta) fit: S = (C + i) / (C - i). The Sheet II pole sits where
    C(s) = i, i.e. where 1/F_2 vanishes.
    """
    F1 = evaluate_F1(model, s_re, s_im)
    s = np.atleast_1d(s_re) + 1j * np.atleast_1d(s_im)
    Cs = C(s)
    F2 = F1 * (Cs + 1j) / (Cs - 1j)
    return np.abs(1.0 / F2) ** 2


def method_B(model, re_range=(0.4, 0.8), im_range=(-0.35, -0.01), res=200, plot=True, verbose=True):
    """
    Refined extraction: hunt for the Sheet II pole in the lower half plane.
    Coarse grid scan of |1/F_2|^2 followed by Nelder-Mead refinement.
    Set plot=False / verbose=False when calling in a loop (e.g. error analysis).
    """
    C, _, _ = fit_cot_delta(model)

    re_vals = np.linspace(re_range[0], re_range[1], res)
    im_vals = np.linspace(im_range[0], im_range[1], res)
    Re, Im = np.meshgrid(re_vals, im_vals)
    inv_F2 = sheet_II_inverse(model, Re.flatten(), Im.flatten(), C).reshape(res, res)

    min_idx = np.unravel_index(np.argmin(inv_F2), inv_F2.shape)
    grid_guess = [Re[min_idx], Im[min_idx]]
    if verbose:
        print(f"Grid minimum: |1/F_2|^2 = {inv_F2[min_idx]:.6e} at "
              f"s = {grid_guess[0]:.4f} {grid_guess[1]:+.4f}i")

    def loss(s_components):
        return sheet_II_inverse(model, s_components[0], s_components[1], C).item()

    res_opt = minimize(loss, grid_guess, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 5000})
    s_pole = complex(res_opt.x[0], res_opt.x[1])
    if verbose:
        print(f"Optimized:    |1/F_2|^2 = {res_opt.fun:.6e} at "
              f"s = {s_pole.real:.4f} {s_pole.imag:+.4f}i")

    sqrt_s = np.sqrt(s_pole)
    m_rho = np.real(sqrt_s)
    gamma_rho = -2.0 * np.imag(sqrt_s)

    if not plot:
        return s_pole, m_rho, gamma_rho

    # Landscape plot of the Sheet II inverse amplitude
    plt.figure(figsize=(8, 6))
    c = plt.pcolormesh(Re, Im, np.sqrt(inv_F2), norm=LogNorm(), cmap='magma', shading='auto')
    plt.plot(s_pole.real, s_pole.imag, 'w*', markersize=15, markeredgecolor='k',
             label=rf'$\sqrt{{s_p}} = {m_rho*1000:.1f} - i\,{gamma_rho/2*1000:.1f}$ MeV')
    plt.colorbar(c, label=r'$|1/F_2(s)|$ (Log Scale)')
    plt.xlabel(r'$\text{Re}(s)$ [GeV$^2$]')
    plt.ylabel(r'$\text{Im}(s)$ [GeV$^2$]')
    plt.title(r'Sheet II Inverse Amplitude $|1/F_2(s)|$')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("Resonance Pole.png", dpi=300)

    return s_pole, m_rho, gamma_rho


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    print("=" * 60)
    print("Method A: 90-degree phase crossing (coarse)")
    print("=" * 60)
    m_A, gamma_A, gamma_A_spread = method_A(model)
    print(f"m_rho     = {m_A * 1000:.2f} MeV")
    print(f"Gamma_rho = {gamma_A * 1000:.2f} MeV  (from phase slope)")
    print(f"Gamma_rho = {gamma_A_spread * 1000:.2f} MeV  (45-135 degree spread, cross-check)")

    print()
    print("=" * 60)
    print("Method B: Sheet II pole search (refined)")
    print("=" * 60)
    s_pole, m_B, gamma_B = method_B(model)
    print(f"s_pole    = {s_pole.real:.6f} {s_pole.imag:+.6f}i GeV^2")
    print(f"m_rho     = {m_B * 1000:.2f} MeV")
    print(f"Gamma_rho = {gamma_B * 1000:.2f} MeV")

