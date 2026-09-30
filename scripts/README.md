# Analysis scripts

Everything that turns a trained checkpoint into a number or a figure. Install the package first (see the top-level [README](../README.md)), then run the scripts from this directory, because they refer to `../weights`, `../dataset` and `../benchmark_dataset` by relative path:

```bash
cd scripts
python <script-name>.py
python benchmark/<script-name>.py   # the benchmarks
python ensemble/<script-name>.py    # the bootstrap uncertainties
```

- Unless stated otherwise a script evaluates the final fit, `weights/final.pt`. To look at another checkpoint, pass its path to `load_model` (or use `--weights` where a script offers it).
- Figures are written to this directory as PNG files; tables as CSV. Both are git-ignored.
- `helper.py` holds what the scripts share: device selection, `load_model`, `load_ensemble` and `phase_degrees`.
- The scripts in `benchmark/` and `ensemble/` are run from this directory as well, not from inside their folder; the relative paths and the output location are then the same as for the others.

In the tables below, "isovector" means the network output itself, `model.forward`, and "with mixing" means `model.forward_mixed` with the ρ–ω term switched on, which is what the e⁺e⁻ data measure.

## Data

| Script | Output | What it shows |
| --- | --- | --- |
| `plot_spacelike_data.py` | `spacelike_dataset_plot.png` | the spacelike data by experiment |
| `plot_timelike_data.py` | `timelike_dataset_plot.png` | the timelike data by experiment |

## Form factor

| Script | Output | What it shows |
| --- | --- | --- |
| `spacelike_form_factor.py` | `Spacelike Region.png` | \|F\|² against the spacelike data |
| `timelike_pure_isovector.py` | `Timelike Isovector.png` | isovector \|F\|² against the τ data of `uncorrected_timelike_dataset.csv` |
| `timelike_interference.py` | `Timelike Interference.png` | \|F\|² with mixing against the e⁺e⁻ data, around the ρ–ω region |
| `continuous_region_form_factor.py` | `Continuous region.png` | isovector \|F\|² across q² = 0, with the spacelike and τ data |
| `continuous_region_form_factor_interference.py` | `Continuous region - interference.png` | the same with mixing for q² > 0 and the e⁺e⁻ data |
| `asymptotic_scaling.py` | `Asymptotic Scaling.png` | Q²\|F(Q²)\| up to Q² = 20 GeV² |
| `phase.py` | `Phase.png` | phase of F on the cut up to 10 GeV² |
| `complex_plane.py` | `Complex Plane.png` | \|F\| over the complex s plane around the cut (surface and heat map) |
| `complex_plane_arg.py` | `Complex Plane Phase.png` | Arg F over the same region |
| `find_zeros.py` | `No Zeros Complex.png` | search for zeros of F in \|Re s\|, \|Im s\| ≤ 20 GeV² (grid scan refined with Nelder–Mead) |

`complex_plane.py` and `complex_plane_arg.py` use Times New Roman if a `times.ttf` is placed in the repository root (the font is not distributed) and fall back to the matplotlib default otherwise.

## Derived quantities

| Script | Output | What it computes |
| --- | --- | --- |
| `chi_sq_fit.py` | printed | χ² of the spacelike and timelike data and their total |
| `dataset_chi_sq_fit.py` | printed, `chi2_by_source.csv` | the same χ² broken down by experiment |
| `calculate_differentials.py` | printed | ⟨r²⟩, ⟨r⁴⟩ and ⟨r⁶⟩ from the derivatives of F at q² = 0 |
| `alpha_omega.py` | printed | the ρ–ω mixing parameters `alpha` and `phi_omega` |
| `hvp_fsr.py` | printed | two-pion contribution to a_μ including final-state radiation, integrated up to s = 0.3969, 1 and 100 GeV² |
| `resonance_pole.py` | printed, `Resonance Pole.png` | ρ mass and width from (A) the 90° crossing of the phase and (B) the pole on the second sheet |
| `dispersion.py` | `Dispersion.png` | Re F of the network against the twice-subtracted dispersion integral over its own Im F |

## Padé parametrisation

A closed-form approximation of the network in the conformal variable w(s), with the ρ pole on the second sheet.

| Script | Output | What it does |
| --- | --- | --- |
| `fit_pade_parametrization.py` | printed, `Pade Parametrization.png` | fits the coefficients to a checkpoint and reports the ρ pole and ⟨r²⟩ of the fit |
| `pade_complex_plane.py` | `Pade Complex Plane.png`, `Pade Complex Plane 3D.png`, `Pade Real Line.png` | \|F\| of the parametrisation against the network over the complex plane and on the real axis |
| `pade_complex_plane_arg.py` | `Pade Complex Plane Phase.png`, `Pade Complex Plane Phase 3D.png` | the same comparison for Arg F |

All three accept `--weights`; see `--help` for the plotting options. The coefficients at the top of `pade_complex_plane.py` are those of `fit_pade_parametrization.py` for `final.pt` and have to be updated by hand for another checkpoint.

## Benchmarks

The scripts in `benchmark/` compare the fit with curves from the literature, described in [`benchmark_dataset/`](../benchmark_dataset/README.md) and [`dataset/`](../dataset/README.md).

| Script | Output | Compared with |
| --- | --- | --- |
| `benchmark/benchmark_asymptotic.py` | `Benchmark Asymptotic Scaling.png` | Q²F(Q²) curves A–D (`benchmark_dataset/A.csv` … `D.csv`) |
| `benchmark/benchmark_form_factor.py` | `Benchmark Form Factor.png` | \|F\|² of `benchmark_dataset/Fig1_a.csv` |
| `benchmark/benchmark_phase.py` | `Benchmark Phase.png` | the phase shift of `dataset/colangelo_phase.csv` |
| `benchmark/benchmark_watson_ablation.py` | `Benchmark Watson Ablation.png` | the same phase shift, for `final.pt` and for `ablation-watson.pt` |
| `benchmark/benchmark_roy.py` | `Benchmark Roy.png`, `benchmark_roy.csv` | the phase shift of `dataset/ananth_phase.csv` (absolute difference) |

## Bootstrap uncertainties

The scripts in `ensemble/` repeat an analysis for every fit of the bootstrap ensemble and report the mean and standard deviation. The fits are read from the `ensemble/` directory of the repository root (`../ensemble/` from here), not from the script folder of the same name. The ensemble is not part of the repository: generate it first with `batch.sh` from the repository root (see the top-level README). The scripts use every `error-no-ablation-*.pt` they find there.

| Script | Output | Quantity |
| --- | --- | --- |
| `ensemble/ensemble_moments.py` | printed | ⟨r²⟩, ⟨r⁴⟩, ⟨r⁶⟩ |
| `ensemble/ensemble_mixing.py` | printed | `alpha`, `phi_omega` |
| `ensemble/ensemble_hvp_fsr.py` | printed | a_μ with final-state radiation up to s = 0.3969 GeV² |
| `ensemble/ensemble_resonance.py` | printed | ρ mass and width, both methods of `resonance_pole.py` |
| `ensemble/ensemble_phase.py` | `Ensemble Phase.png` | phase with its 1σ band |
| `ensemble/ensemble_spacelike.py` | `Ensemble Spacelike Region.png` | spacelike \|F\|² with its 1σ band |
| `ensemble/ensemble_timelike_isovector.py` | `Ensemble Timelike Isovector.png` | isovector timelike \|F\|² with its 1σ band, against the τ data |
| `ensemble/ensemble_timelike_interference.py` | `Ensemble Timelike Interference.png` | timelike \|F\|² with mixing and its 1σ band, against the e⁺e⁻ data |

## Loss-weight sweep

`lhs_sweep.py` builds the Latin-hypercube design over the eleven loss weights (each within a factor of two of its baseline) and trains it.

```bash
python lhs_sweep.py                        # print the 64 weight configurations
python lhs_sweep.py --train                # train all of them into ../weights/lhs/
python lhs_sweep.py --train --trials 0-15  # or only a slice
```

Trials whose checkpoint already exists are skipped unless `--overwrite` is given, so with the published `weights/lhs/` in place `--train` does nothing. Each trial starts from `weights/final.pt`; see `--help` for the remaining options.
