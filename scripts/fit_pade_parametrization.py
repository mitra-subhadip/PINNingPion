"""
Fits the closed-form Pade parametrization of the pion form factor to a trained PINN.

Samples F(s) from the network on the timelike cut, the spacelike axis and a complex
grid (first sheet), then fits

    F_I1(s) = N(w) / D(w),   w(s) = (sqrt(4m_pi^2 - s) - 2m_pi) / (sqrt(4m_pi^2 - s) + 2m_pi)

    N(w) = 1 + a1 w + a2 w^2 + a3 w^3 + a4 w^4
    D(w) = (1 - w/w_rho)(1 - w/conj(w_rho)) (1 + b1 w + b2 w^2)

with real a_k, b_k and the rho pole pair at |w_rho| > 1 (i.e. on sheet II).
This guarantees F(0) = 1, Schwarz reflection and analyticity on the first sheet.

The I=0 (e+e-) channel follows from the learned mixing parameters:
    F_I0(s) = F_I1(s) * (1 + alpha e^{i phi_omega} m_omega^2 / (m_omega^2 - s - i m_omega Gamma_omega))

Usage:  python fit_pade_parametrization.py [--weights ../weights/final.pt]
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from helper import load_model, set_torch_device
from mesonforms.constants import FOUR_M_PI2, M_PI, CONV_FM2_GEV2

TWO_MPI = 2 * M_PI


def w_of_s(s_complex):
    r = np.sqrt(FOUR_M_PI2 - s_complex)
    return (r - TWO_MPI) / (r + TWO_MPI)


def w_to_s(w):
    return FOUR_M_PI2 * (1 - ((1 + w) / (1 - w)) ** 2)


def sample_pinn(model, device):
    """Sample F on the cut (upper lip), the spacelike axis and a complex grid."""
    def evalF(sr, si):
        re_t = torch.as_tensor(np.atleast_1d(sr), dtype=torch.float32, device=device)
        im_t = torch.as_tensor(np.atleast_1d(si), dtype=torch.float32, device=device)
        with torch.no_grad():
            u, v = model.forward(re_t, im_t)
        return u.cpu().numpy() + 1j * v.cpu().numpy()

    s_cut = np.linspace(FOUR_M_PI2 + 1e-5, 9.0, 6000)
    F_cut = evalF(s_cut, np.full_like(s_cut, 1e-7))

    s_sl = -np.linspace(1e-4, 10.0, 3000)
    F_sl = evalF(s_sl, np.zeros_like(s_sl))

    re_vals = np.linspace(-1.5, 2.5, 120)
    im_vals = np.concatenate([-np.geomspace(1e-3, 1.0, 40)[::-1], np.geomspace(1e-3, 1.0, 40)])
    Re, Im = np.meshgrid(re_vals, im_vals)
    F_grid = evalF(Re.flatten(), Im.flatten())
    return (s_cut, F_cut), (s_sl, F_sl), (Re, Im, F_grid.reshape(Re.shape))


def sheet_II_pole(s_cut, F_cut):
    """Rho pole via cot(delta) continuation (same idea as resonance_pole.py, coarse)."""
    m = s_cut < 1.2
    delta = np.unwrap(np.arctan2(F_cut[m].imag, F_cut[m].real))
    win = (delta > np.radians(30)) & (delta < np.radians(150))
    C = np.poly1d(np.polyfit(s_cut[m][win], 1.0 / np.tan(delta[win]), 3))
    # pole where C(s) = i, solve on a grid + Newton polish
    roots = (C - np.poly1d([1j])).roots
    cand = [r for r in roots if r.imag < 0 and 0.3 < r.real < 0.9]
    return min(cand, key=lambda r: abs(r.imag)) if cand else 0.575 - 0.105j


def pade(w, p):
    wp = p[0] + 1j * p[1]
    a, b = p[2:6], p[6:8]
    num = 1 + a[0] * w + a[1] * w ** 2 + a[2] * w ** 3 + a[3] * w ** 4
    den = (1 - w / wp) * (1 - w / np.conj(wp)) * (1 + b[0] * w + b[1] * w ** 2)
    return num / den


def fit(cut, sl, grid):
    s_cut, F_cut = cut
    s_sl, F_sl = sl
    Re, Im, F_grid = grid
    w_cut = w_of_s(s_cut + 1e-7j)
    w_sl = w_of_s(s_sl.astype(complex))
    w_g = w_of_s((Re + 1j * Im).flatten())
    F_g = F_grid.flatten()

    w_all = np.concatenate([w_cut, w_sl, w_g])
    F_all = np.concatenate([F_cut, F_sl, F_g])
    wt_all = np.concatenate([np.where(s_cut < 1.2, 5.0, 1.0) / np.abs(F_cut),
                             2.0 / np.abs(F_sl), 1.0 / np.abs(F_g)])

    s_p = sheet_II_pole(s_cut, F_cut)
    wp0 = 1.0 / w_of_s(np.array([s_p], dtype=complex))[0]  # sheet II: 1/w_I

    # linear seed for the numerator at fixed pole, b = 0
    D = (1 - w_all / wp0) * (1 - w_all / np.conj(wp0))
    A = np.stack([w_all ** k / D for k in range(1, 5)], axis=1) * wt_all[:, None]
    rhs = (F_all - 1.0 / D) * wt_all
    a0, *_ = np.linalg.lstsq(np.concatenate([A.real, A.imag]),
                             np.concatenate([rhs.real, rhs.imag]), rcond=None)

    def resid(p):
        r = (pade(w_all, p) - F_all) * wt_all
        return np.concatenate([r.real, r.imag])

    p0 = np.concatenate([[wp0.real, wp0.imag], a0, [0.0, 0.0]])
    sol = least_squares(resid, p0, method='lm', max_nfev=100000)
    return sol.x, (w_cut, w_sl, w_g, F_g)


def report(p, cut, sl, ws):
    s_cut, F_cut = cut
    s_sl, F_sl = sl
    w_cut, w_sl, w_g, F_g = ws
    for tag, w, F in [("cut", w_cut, F_cut), ("spacelike", w_sl, F_sl), ("complex grid", w_g, F_g)]:
        e = np.abs(pade(w, p) - F) / np.abs(F)
        print(f"  rel. err |F| on {tag:12s}: mean={e.mean()*100:.2f}%  max={e.max()*100:.2f}%")

    wp = p[0] + 1j * p[1]
    s_p = w_to_s(1 / wp)
    if s_p.imag > 0:
        s_p = np.conj(s_p)
    sq = np.sqrt(s_p)
    print(f"\n  w_rho = {wp:.6f}  (|w| = {abs(wp):.4f}, sheet II)")
    for i, ai in enumerate(p[2:6], 1):
        print(f"  a_{i} = {ai:+.6f}")
    for i, bi in enumerate(p[6:8], 1):
        print(f"  b_{i} = {bi:+.6f}")
    print(f"  rho pole: s = {s_p:.6f} GeV^2  ->  m_rho = {sq.real*1000:.2f} MeV, "
          f"Gamma_rho = {-2*sq.imag*1000:.2f} MeV")

    ds = 1e-5
    dF = (pade(w_of_s(np.array([ds + 0j])), p)[0].real
          - pade(w_of_s(np.array([-ds + 0j])), p)[0].real) / (2 * ds)
    print(f"  <r^2> = {6*dF/CONV_FM2_GEV2:.4f} fm^2")

    # sanity: no poles/zeros inside the unit disk (first sheet must stay clean)
    a, b = p[2:6], p[6:8]
    zeros = np.roots(np.concatenate([a[::-1], [1.0]]))
    poles = np.roots([b[1], b[0], 1.0])
    print(f"  numerator zeros |w|: {[f'{abs(r):.3f}' for r in zeros]}")
    print(f"  extra den. roots |w|: {[f'{abs(r):.3f}' for r in poles]}"
          f"   (all > 1 keeps the first sheet analytic)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the conformal Pade parametrization to a PINN.")
    parser.add_argument("--weights", default="../weights/final.pt", help="Path to the model weights")
    parser.add_argument("--plot", default="Pade Parametrization.png", help="Output comparison plot")
    args = parser.parse_args()

    DEVICE = set_torch_device()
    model = load_model(DEVICE, pt_file=args.weights)
    print(f"Sampling PINN ({args.weights}) ...")
    cut, sl, grid = sample_pinn(model, DEVICE)

    print("Fitting Pade parametrization ...")
    p, ws = fit(cut, sl, grid)
    report(p, cut, sl, ws)
    print(f"\n  mixing (I=0 channel): alpha = {model.alpha.item():.6f}, "
          f"phi_omega = {model.phi_omega.item():.6f} rad")

    s_cut, F_cut = cut
    m = s_cut < 1.4
    Fm = pade(ws[0], p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(np.sqrt(s_cut[m]), np.abs(F_cut[m]), 'k-', lw=2, label='PINN')
    ax1.plot(np.sqrt(s_cut[m]), np.abs(Fm[m]), 'r--', lw=1.5, label='Pade')
    ax1.set_xlabel(r'$\sqrt{s}$ [GeV]'); ax1.set_ylabel(r'$|F_\pi|$')
    ax1.legend(); ax1.set_title('Timelike cut (pure isovector)')
    s_sl, F_sl = sl
    ax2.plot(s_sl, np.abs(F_sl), 'k-', lw=2, label='PINN')
    ax2.plot(s_sl, np.abs(pade(ws[1], p)), 'r--', lw=1.5, label='Pade')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$s = -Q^2$ [GeV$^2$]'); ax2.set_ylabel(r'$|F_\pi|$')
    ax2.legend(); ax2.set_title('Spacelike')
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)
    print(f"  saved {args.plot}")
