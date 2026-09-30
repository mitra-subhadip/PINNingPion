## Files

| File | Columns | Contents |
| --- | --- | --- |
| `spacelike_dataset.csv` | `q2` (Q² = −q² in GeV²), `Fpi_sq`, `error`, `Source` | \|F<sub>π</sub>\|² and its uncertainty in the spacelike region |
| `timelike_dataset.csv` | `q2` (GeV²), `Fpi_sq`, `error`, `Source`, `I` | \|F<sub>π</sub>\|² and its uncertainty in the timelike region. `I = 1` marks τ-decay data (pure isovector), `I = 0` marks e⁺e⁻ data. |
| `uncorrected_timelike_dataset.csv` | as above | Variant of `timelike_dataset.csv` with the uncorrected τ data (BELLE, CLEO); its BABAR and KLOE rows differ as well. |
| `colangelo_phase.csv` | `q` (√s in GeV), `phase` (degrees) | ππ P-wave phase shift δ₁¹ from https://arxiv.org/abs/hep-ph/0103088. Target of the Watson loss. |
| `ananth_phase.csv` | `q` (√s in GeV), `phase` (degrees) | δ₁¹ read off Fig. 13 of https://arxiv.org/abs/hep-ph/0005297 with PlotDigitizer. Used only as a benchmark. |

The form-factor data are taken from the following papers:

## DOI for Spacelike region datapoints
Check the source column in `spacelike_dataset.csv` file

**JLab:** https://doi.org/10.1103/PhysRevC.78.045202

**NA7:** https://doi.org/10.1016/0550-3213(86)90437-2

**Fermilab:** (Table II) https://doi.org/10.17182/hepdata.26467.v1

**CESR:** (Table IV) https://doi.org/10.1103/PhysRevD.13.25

**CEA:** https://doi.org/10.1103/PhysRevD.8.92 (also Table V of CESR paper)

**CORNELL:** https://doi.org/10.1103/PhysRevD.9.1229 (also Table VI of CESR paper)


## DOI for Timelike region datapoints
Check the source column in `timelike_dataset.csv` file

**BELLE:** https://doi.org/10.1103/PhysRevD.78.072006

**CLEO:** https://doi.org/10.1103/PhysRevD.61.112002

**CMD-2:** (Table I) https://doi.org/10.17182/hepdata.41782

**CMD-3:** https://doi.org/10.1103/PhysRevLett.132.231903

**VEPP-2M:** https://doi.org/10.17182/hepdata.27474.v1

**DM-2:** https://doi.org/10.17182/hepdata.29829.v1

**CLEO-c:** https://doi.org/10.17182/hepdata.130771.v1

**KLOE:** https://doi.org/10.17182/hepdata.96268.v1

**OLYA:** (Table I) https://doi.org/10.17182/hepdata.6886.v2

**FRASCATI:** https://doi.org/10.17182/hepdata.37445.v1

**BaBar:** https://www.hepdata.net/record/152373

**BESIII:** https://doi.org/10.17182/hepdata.73898.v4

**SND:** https://doi.org/10.17182/hepdata.114983.v1


### To convert CLEO data into pion form factor, the following methodology was used:

1. According to the paper, the spectral function averaged over the $i$-th bin of central mass $M_i$ and width $\Delta M_i$ is given by Equation 15:
$$\overline{v}^{\pi\pi}(M_i) = \frac{B_{\pi\pi^0}}{B_e} \frac{M_\tau^8}{12\pi |V_{ud}|^2 S_{EW}} \frac{1}{M_i (M_\tau^2 - M_i^2)^2 (M_\tau^2 + 2M_i^2)} \frac{N_i/N}{\Delta M_i}$$
Where $N_i/N$ is the normalized entry count for the bin (given in Table IV as "Entries $\times 10^{-4}$").Next, the relationship between the spectral function and the squared form factor is defined by reversing Equation 4:
$$|F_\pi(M_i)|^2 = 12\pi \cdot \overline{v}^{\pi\pi}(M_i) \left( \frac{M_i}{2p_\pi} \right)^3$$
Where the phase space momentum $2p_\pi$ for the unequal-mass $\pi^- \pi^0$ system is dynamically calculated as:
$$2p_\pi = \frac{\sqrt{[M_i^2 - (m_{\pi^-} + m_{\pi^0})^2][M_i^2 - (m_{\pi^-} - m_{\pi^0})^2]}}{M_i}$$

2. All constants are taken directly from the text and Table VI of the paper:
$B_{\pi\pi^0} = 0.2532$
$B_e = 0.1781$
$M_\tau = 1.77705 \text{ GeV}$
$|V_{ud}| = 0.9752$
$S_{EW} = 1.0194$ (the overall radiative correction ratio $S_{EW}^{\pi\pi}/S_{EW}^e$)
$m_{\pi^-} = 0.13957 \text{ GeV}$$m_{\pi^0} = 0.13498 \text{ GeV}$


### To convert BaBar data into pion form factor, the following methodology was used:

$$\sigma(s) = \frac{\pi \alpha^2}{3s} \beta^3 |F_\pi(s)|^2 \times (\hbar c)^2$$

$s$ is the squared center-of-mass energy ($s = q^2$)

$\alpha$ is the fine-structure constant ($\approx 1/137.036$)

$\beta$ is the kinematic phase-space velocity of the pion, defined as $\sqrt{1 - 4M_\pi^2 / s}$

$(\hbar c)^2$ is the unit conversion factor to nanobarns ($\approx 0.389379 \times 10^6 \text{ nb GeV}^2$).

By inverting this equation, we extract the form factor:
$$|F_\pi(s)|^2 = \frac{3s}{\pi \alpha^2 \beta^3 (\hbar c)^2} \sigma(s)$$

The statistical errors were propagated using fractional uncertainties: $\frac{\delta |F_\pi|^2}{|F_\pi|^2} = \frac{\delta \sigma}{\sigma}$.
