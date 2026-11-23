---
layout: project_page
permalink: /

# ------------------------------------------------------------------
# Page metadata
# ------------------------------------------------------------------
# Plain-text title (browser tab, link previews)
title: "PINNing the pion: A Geometric Deep Learning approach for Fπ(s) and the (g−2)μ hadronic contribution"
# Title shown on the page (MathJax allowed)
title_html: 'PINNing the pion: A Geometric Deep Learning approach for \(F_{\pi}(s)\) and the \((g-2)_{\mu}\) hadronic contribution'
description: "A physics-informed neural network in the conformal unit disk that reconstructs the pion form factor from S-matrix principles and data, with model-independent estimates of the charge radius, the ρ(770) pole and the two-pion contribution to the muon g−2."
og_image: /static/images/cover_loss_landscape.png

# ------------------------------------------------------------------
# Authors — `url` is optional; `affils` refers to the ids below
# ------------------------------------------------------------------
authors:
  - name: Mayank Goel
    affils: [1]
    url: https://orcid.org/0009-0003-6204-3663
  - name: Subhadip Mitra
    affils: [1, 2]
    url: https://sites.google.com/site/subhadipmitra/
  - name: Monalisa Patra
    affils: [3]
    url: https://orcid.org/0000-0002-3242-0267

affiliations:
  - id: 1
    name: Center for Computational Natural Sciences and Bioinformatics, IIIT Hyderabad
  - id: 2
    name: Center for Quantum Science and Technology, IIIT Hyderabad
  - id: 3
    name: iHub-Data, IIIT Hyderabad

# Optional one-liners under the authors, e.g. acceptance news
announcements:
  # - "🎉 Accepted at ..."

# ------------------------------------------------------------------
# Buttons. Leave `url` empty to show the `note` as a greyed-out
# button instead, or delete the entry to hide it.
# Icons: https://fontawesome.com/v5/search  ·  ai ai-arxiv (Academicons)
# ------------------------------------------------------------------
links:
  - label: Paper
    url: https://arxiv.org/pdf/2609.40008
    icon: fas fa-file-pdf
  - label: arXiv
    url: https://arxiv.org/abs/2609.40008
    icon: ai ai-arxiv
  - label: Code
    url: https://github.com/mitra-subhadip/PINNingPion
    icon: fab fa-github

# ------------------------------------------------------------------
# Teaser figure under the title
# ------------------------------------------------------------------
teaser:
  image: /static/images/cover_loss_landscape.png
  width: 85%
  caption: >-
    Filter-normalised loss landscape around the optimised minimum $$\theta^*$$.
    Working in the bounded conformal plane keeps the curvature of this landscape
    finite, so gradient descent can reach the minimum.

abstract: |
  The pion electromagnetic form factor $$F_\pi(s)$$ is a fundamental quantity in hadronic physics. Extracting the form factor through traditional phenomenological curve-fitting models introduces model dependence, kinematic inconsistencies, and unphysical artefacts. We introduce a Physics-Informed Neural Network (PINN) framework that works in the unit disk space (conformal $$z$$-space) instead of the traditional complex $$q^2$$ plane ($$s$$-space). Working in the conformal space has optimisation benefits. We establish certain convergence properties for gradient descent. We construct $$F_\pi(z)$$ in the unit-disk space in a first-principles manner jointly across spacelike and timelike domains, without using any explicit functional form. The charge normalisation and Schwarz reflection principle constraints are enforced in our neural network by construction, while analyticity, dispersion relations, Watson's theorem, and perturbative QCD asymptotics enter through the loss function. We find that the S-matrix principles strongly constrain the form factor in the spacelike region, as the network can closely predict its behaviour there using only timelike data. Besides $$e^+e^-$$ scattering data, we also incorporate $$\tau$$-decay data through a switch that isolates the pure isovector form factor natively, bypassing model-dependent isospin-breaking pre-corrections. The network organically yields an interior zero-free form factor. Our best estimates yield a pion charge radius of $$\langle r_\pi^2 \rangle = 0.435 \pm 0.008_{\rm stat} \pm 0.007_{\rm cali}$$ fm², second-sheet pole parameters $$m_\rho^{\rm pole} = 761.72 \pm 1.04_{\rm stat} \pm 0.57_{\rm cali}$$ MeV and $$\Gamma_\rho^{\rm pole} = 135.99 \pm 1.20_{\rm stat} \pm 1.12_{\rm cali}$$ MeV, and a two-pion contribution to the muon anomalous magnetic moment of $$a_\mu^{\pi\pi} = (506.48 \pm 2.02_{\rm stat} \pm 1.70_{\rm cali}) \times 10^{-10}$$.

# ------------------------------------------------------------------
# Headline numbers (cards under the abstract). Use \( ... \) for math.
# ------------------------------------------------------------------
highlights:
  - label: 'Pion charge radius \(\langle r_\pi^2\rangle\)'
    value: 0.435 ± 0.008 ± 0.007 fm²
    note: ± stat ± calibration
  - label: 'ρ(770) pole \(m_\rho,\ \Gamma_\rho\)'
    value: 761.72 ± 1.04 ± 0.57 MeV<br>135.99 ± 1.20 ± 1.12 MeV
    note: second Riemann sheet
  - label: 'Two-pion HVP \(a_\mu^{\pi\pi}\)'
    value: (506.48 ± 2.02 ± 1.70) × 10⁻¹⁰
    note: full energy range

bibtex: |
  @article{Goel:2026evy,
    title         = {PINNing the pion: A Geometric Deep Learning approach for $F_{\pi}(s)$ and the $(g-2)_{\mu}$ hadronic contribution},
    author        = {Goel, Mayank and Mitra, Subhadip and Patra, Monalisa},
    month         = {9},
    year          = {2026},
    eprint        = {2609.40008},
    archivePrefix = {arXiv},
    primaryClass  = {hep-ph},
    url           = {https://arxiv.org/abs/2609.40008}
  }
---

<!--
  Writing tips
  - Math: wrap in $$...$$, both inline and on its own line (Kramdown → MathJax).
  - Figures: {% raw %}{% include figure.html src="/static/images/NAME.png" caption="..." width="80%" %}{% endraw %}
    A dashed placeholder shows until the file exists.
  - Side-by-side figures: wrap two includes in <div class="figure-row"> ... </div>
-->

## Why a conformal plane?

The pion form factor $$F_\pi(s)$$ links the low-energy QCD vacuum to the two-pion contribution that dominates the hadronic vacuum polarisation in the muon $$g-2$$. Reconstructing it across both the spacelike ($$s<0$$) and timelike ($$s > 4m_\pi^2$$) regions while respecting every S-matrix constraint is hard for phenomenological models, and, as we show, also for neural networks.

Trained directly on the physical variable $$s$$, a network faces an **unbounded input**. With periodic activations, parameter gradients grow with $$\lvert s\rvert$$ and the curvature of the loss landscape explodes. With saturating activations such as tanh, gradients vanish instead. Either way, the minimum eigenvalue of the neural tangent kernel (NTK) collapses in the deep spacelike region: the network suffers *spectral starvation* and cannot learn the high-$$\lvert q^2\rvert$$ behaviour that pQCD dictates.

The fix is the same map dispersive physics already uses. The conformal transformation

$$
z(s) = \frac{\sqrt{4m_\pi^2 - s} - 2m_\pi}{\sqrt{4m_\pi^2 - s} + 2m_\pi}
$$

compactifies the entire cut $$s$$-plane onto the unit disk. With bounded inputs, the Hessian of the loss is uniformly bounded, so gradient descent converges, and the NTK stays positive definite, so no kinematic region becomes a dead zone for learning. **The geometry that enforces analyticity is the geometry that stabilises optimisation.**

{% include figure.html src="/static/images/fig1_conformal_map.png" width="75%" caption="The conformal map $$z(s)$$ sends the physical sheet to the unit disk. The branch cut $$[4m_\pi^2, \infty)$$ maps to the boundary circle, and the $$\rho(770)$$ pole sits on the second sheet outside the disk, so $$F_\pi$$ is holomorphic for $$\lvert z\rvert < 1$$." %}

## Physics built in, not fitted

Two constraints are enforced **exactly by the architecture**:

- **Charge normalisation.** The ansatz $$F_\pi(z) = 1 + s(z)\,\mathcal{N}(z)$$ guarantees $$F_\pi(0) = 1$$ for any network weights.
- **Schwarz reflection.** The network sees only $$\lvert\mathrm{Im}\,z\rvert$$, and the imaginary part of its output is multiplied by $$\mathrm{sgn}(\mathrm{Im}\,z)$$, so $$F_\pi(s^*) = F_\pi^*(s)$$ holds everywhere.

The remaining S-matrix and QCD constraints enter through the **loss functional**, alongside the data:

| Loss term | What it enforces |
|---|---|
| Analyticity | Cauchy–Riemann conditions in the disk, sampled stochastically every epoch |
| Dispersion | Twice-subtracted dispersion relation, evaluated on the conformal boundary |
| Moments | Low-energy derivatives agree with spectral sum rules (counters spectral bias) |
| Positivity | $$\mathrm{Im}\,F_\pi \ge 0$$ on the cut (unitarity) |
| Watson | Phase locked to the $$\pi\pi$$ P-wave phase shift $$\delta_1^1$$ below 1 GeV² |
| Monotonicity | Phase never decreases along the cut |
| pQCD | NLO $$1/Q^2$$ scaling in the deep spacelike region |
| Timelike asymptotic | $$\mathrm{Re}\,F_\pi \sim 1/s$$ as $$s \to \infty$$ |

{% include figure.html src="/static/images/fig2_architecture.png" caption="**The PINN in the conformal plane.** The network takes $$\mathrm{Re}\,z$$, $$\mathrm{Im}\,z$$ and a channel switch ($$e^+e^-$$ vs. $$\tau$$ decay) through four hidden layers of 128 adaptive-sine neurons. Its output is combined with the ansatz $$F_\pi(z) = 1 + s(z)\,\mathcal{N}(z)$$, which hard-wires $$F_\pi(0)=1$$." %}

No functional form is assumed for $$\mathrm{Im}\,F_\pi$$. A **channel switch** sets the $$\rho$$–$$\omega$$ mixing strength $$\alpha_{\rho-\omega}$$ to zero for $$\tau$$-decay data and leaves it free for $$e^+e^-$$ data, so one network learns the pure isovector form factor from both, without point-by-point isospin-breaking corrections.

### Data

We train on 19 datasets: spacelike $$e\pi$$ scattering and pion electroproduction (NA7, Fermilab, CEA, Cornell, JLab), timelike $$e^+e^- \to \pi^+\pi^-(\gamma)$$ (BaBar, CMD-2, CMD-3, KLOE, BESIII, SND, OLYA, DM2, ADONE, CLEO-c), and $$\tau^- \to \pi^-\pi^0\nu_\tau$$ (Belle, CLEO). Our baseline, **Set A**, uses spacelike plus $$e^+e^-$$ data without CMD-3; Sets B–D probe CMD-3, $$\tau$$ data and their combination.

## Results

### A zero-free form factor, without asking for one

The reconstructed form factor obeys Schwarz reflection across the complex plane and shows the $$\rho(770)$$ peak on the cut, with no spurious poles. No loss term asks for the absence of zeros, yet none appear in the interior: independent, data-driven support for the zero-free hypothesis that dispersive analyses otherwise have to impose.

{% include figure.html src="/static/images/fig3_analytic_structure.png" caption="Top: $$\lvert F_\pi(s)\rvert$$ on the first sheet, peaking at the $$\rho(770)$$ on the cut (red). Bottom: its phase, which flips sign across the cut as Schwarz reflection requires." %}

{% include figure.html src="/static/images/fig4_zero_free.png" width="60%" caption="$$\lvert F_\pi(s)\rvert$$ over a wide kinematic range: no zeros in the interior, and a smooth $$1/s$$ fall-off." %}

### One curve from spacelike to timelike

A single network reproduces the spacelike measurements, the $$\rho$$ resonance and the sub-per-cent $$\rho$$–$$\omega$$ interference ripple, with the dispersion relation tying the two regions together.

{% include figure.html src="/static/images/fig6_form_factor.png" width="85%" caption="$$\lvert F_\pi(s)\rvert^2$$ across both kinematic regions, with the experimental data (Set A)." %}

<div class="figure-row">
{% include figure.html src="/static/images/fig7_timelike.png" caption="Timelike region: the $$\rho$$ resonance and the $$\rho$$–$$\omega$$ dip-bump pattern." %}
{% include figure.html src="/static/images/fig8_spacelike.png" caption="Spacelike region: smooth, monotonic damping through the electroproduction data." %}
</div>

<div class="callout" markdown="1">
**The spacelike region from timelike data alone.** Training without any spacelike data ($$\lambda_{\rm spacelike} = 0$$) leaves the result almost unchanged: $$\langle r_\pi^2\rangle = 0.436$$ fm² and $$a_\mu^{\pi\pi} = 505.68 \times 10^{-10}$$, against 0.435 fm² and 506.48 with it. The S-matrix constraints carry the timelike information across to $$s < 0$$.
</div>

At large $$Q^2$$, the prediction bridges the non-perturbative and perturbative regimes up to 20 GeV², closely tracking the Dyson–Schwinger benchmark and staying well below the monopole (VMD) bound.

{% include figure.html src="/static/images/fig9_asymptotics.png" width="80%" caption="$$Q^2\lvert F_\pi(Q^2)\rvert$$ in the spacelike domain: PINN (black) against DSE (A), monopole (B) and two pQCD estimates (C, D)." %}

### Phase, Watson's theorem and the $$\rho$$ pole

The extracted phase tracks the Roy-equation solution with no visible offset, even though the network never sees the low-energy scattering lengths those solutions are anchored to. Switching off the Watson loss makes the phase drift away, increasingly with energy.

{% include figure.html src="/static/images/fig10_phase.png" width="80%" caption="Phase of $$F_\pi$$ (black) against the Roy-equation solution (orange). Blue dotted: the same network trained without the Watson loss." %}

Continuing to the second Riemann sheet via elastic unitarity, a Newton–Raphson search finds the $$\rho$$ pole at $$s_{\rm pole} = (0.5756 - 0.1036\,i)$$ GeV², giving $$m_\rho^{\rm pole} = 761.72 \pm 1.04 \pm 0.57$$ MeV and $$\Gamma_\rho^{\rm pole} = 135.99 \pm 1.20 \pm 1.12$$ MeV. The mass agrees with the PDG pole to within 1%; the width sits about 4–8% below it.

### Charge radius and higher moments

Uncertainties are quoted as $$\pm\sigma_{\rm stat} \pm \sigma_{\rm cali}$$: a 500-replica bootstrap over the data, and a 64-point Latin-hypercube scan of the loss weights.

| Observable | Previous estimates | This work |
|---|---|---|
| $$\langle r_\pi^2\rangle$$ (fm²) | Lattice: 0.42 ± 0.02 · PDG: 0.434 ± 0.005 | **0.435 ± 0.008 ± 0.007** |
| $$\langle r_\pi^4\rangle$$ (fm⁴) | 2-loop ChPT: 0.7 ± 0.11 · Dispersive: 0.68–0.72 | **0.748 ± 0.057 ± 0.048** |
| $$\langle r_\pi^6\rangle$$ (fm⁶) | Dispersive: 2.95–3.11 · Truong: 2.89 ± 0.12 | **4.01 ± 0.95 ± 0.90** |

### Muon $$g-2$$: the two-pion contribution

| Energy range (GeV) | Previous estimates ($$\times 10^{-10}$$) | This work ($$\times 10^{-10}$$) |
|---|---|---|
| $$[2m_\pi,\ 0.63]$$ | 133.26 ± 0.72 (Ananthanarayan et al.) · 130.47 ± 0.64 (Colangelo et al.) | **130.66 ± 0.67 ± 1.20** |
| $$[2m_\pi,\ 1.0]$$ | 495.0 ± 2.58 (Colangelo et al.) | **496.37 ± 2.08 ± 1.70** |
| $$[2m_\pi,\ \infty)$$ | 507.9 ± 3.29 (Theory Initiative 2020) · 513.2 ± 3.8 (Davier et al.) | **506.48 ± 2.02 ± 1.70** |

The total uncertainty on the full-range result, ±2.64, is smaller than in traditional evaluations (±3.29 to ±3.8), because the physics losses suppress point-to-point fluctuations that data-driven integrations propagate through the kernel.

### Testing the tension in the data

Training on different subsets turns the network into an analytical arbiter. CMD-3 alone pushes $$a_\mu^{\pi\pi}$$ up by about $$26 \times 10^{-10}$$, and the network has to distort its high-energy phase to accommodate it.

| Training set | $$a_\mu^{\pi\pi}\,[2m_\pi, \infty) \times 10^{10}$$ | Phase deviation at (1.15 GeV)² |
|---|---|---|
| A: spacelike + $$e^+e^-$$ without CMD-3 | 506.48 ± 2.02 ± 1.70 | 1.72 ± 0.15 ± 0.26° |
| B: spacelike + CMD-3 only | 532.27 ± 3.82 ± 1.80 | 5.08 ± 0.33 ± 0.47° |
| C<sub>raw</sub>: spacelike + raw $$\tau$$ (channel switch) | 512.79 ± 5.20 ± 1.74 | 6.65 ± 0.71 ± 0.88° |
| C<sub>R<sub>IB</sub></sub>: spacelike + $$\tau$$ with isospin correction | 502.83 ± 5.14 ± 1.71 | 6.28 ± 1.17 ± 2.01° |
| D: all data combined | 515.30 ± 1.73 ± 1.73 | 3.46 ± 0.47 ± 0.30° |

<div class="callout" markdown="1">
**Ablations at a glance.** Watson's theorem is the single most important physics loss: removing it inflates $$\langle r_\pi^2\rangle$$ by ~25% and shifts the phase by up to 11°. Removing *all* physics losses still fits the data points, but the moments $$\langle r_\pi^4\rangle$$ and $$\langle r_\pi^6\rangle$$ turn negative, and the phase misses by more than 40°.
</div>

{% include figure.html src="/static/images/fig11_tau.png" caption="$$\tau$$-decay reconstructions: (a) raw data with the channel switch isolating $$F_\pi^{I=1}$$; (b) data pre-corrected with $$R_{\rm IB}(s)$$, which slightly underfits the peak." %}

## A closed-form form factor

For use in other calculations, we compress the network into a [4/4] conformal Padé approximant that follows it closely near the real axis and the $$\rho$$ pole:

$$
F_\pi(s) = \frac{1 + a_1 z + a_2 z^2 + a_3 z^3 + a_4 z^4}{\left(1 - z/z_\rho\right)\left(1 - z/z_\rho^*\right)\left(1 + b_1 z + b_2 z^2\right)}, \qquad z = z(s)
$$

| $$a_1$$ | $$a_2$$ | $$a_3$$ | $$a_4$$ | $$b_1$$ | $$b_2$$ | $$z_\rho$$ |
|---|---|---|---|---|---|---|
| −3.527647 | 4.755645 | −2.937709 | 0.707660 | −1.554048 | 0.758975 | 0.790353 − 0.728033 *i* |

All denominator roots lie outside the unit disk, so the approximant is holomorphic for $$\lvert z\rvert \le 1$$. Its pole alone gives $$m_\rho = 760.7$$ MeV and $$\Gamma_\rho = 140.1$$ MeV, an independent cross-check of the direct extraction.

{% include figure.html src="/static/images/fig13_pade.png" width="80%" caption="$$\lvert F_\pi(s)\rvert$$ from the full PINN (black) and the Padé approximant (red dashed): indistinguishable through the $$\rho(770)$$ peak." %}
