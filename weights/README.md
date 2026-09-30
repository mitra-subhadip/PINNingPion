# Trained checkpoints

Every `.pt` file is a checkpoint of the same network, `PINN_Fpi(hidden_units=128, hidden_layers=4)` (67 204 parameters), as written by `PINNTrainer.save_checkpoint`: a dictionary with

- `model_state_dict`: the network weights and the two ρ–ω mixing parameters `alpha` and `phi_omega`,
- `optimizer_state_dict`: the state of the Adam optimiser at the best epoch.

## Loading a checkpoint

From the `scripts/` directory:

```python
from helper import load_model, set_torch_device

device = set_torch_device()
model = load_model(device)                                   # weights/final.pt
model = load_model(device, "../weights/ablation-watson.pt")  # any other checkpoint

u, v = model.forward(q2)              # Re F, Im F at real q^2 (GeV^2), pure isovector
u, v = model.forward(q2_re, q2_im)    # complex q^2
u, v = model.forward_mixed(q2, mask)  # mask = 1 adds the rho-omega mixing term (e+e- channel)
```

All checkpoints except `final.pt` were saved from a GPU (CUDA, or MPS for some of the earlier ones). `load_model` and `main.py` move them to whatever device is available; with a bare `torch.load` pass `map_location`.

## Final fit

| File | Description |
| --- | --- |
| `final.pt` | The fit presented in the paper and the default of every script. Trained on the spacelike data and the e⁺e⁻ timelike data without CMD-3, with all loss terms at their baseline weights. |

The baseline weights are the `WEIGHTS` dictionary in [`main.py`](../main.py):

| Term | Weight | Constraint |
| --- | --- | --- |
| `spacelike_data` | 800 | fit to the spacelike data |
| `timelike_data` | 1000 | fit to the timelike data |
| `analyticity` | 400 | Cauchy–Riemann equations inside the conformal disk |
| `watson` | 300 | Watson's theorem: phase equal to the ππ P-wave phase shift in the elastic region |
| `dispersion` | 100 | twice-subtracted dispersion relation |
| `moments` | 400 | sum rules for the first three derivatives at q² = 0 |
| `asymptotics` | 300 | F and its derivative vanish at infinity |
| `pqcd` | 50 | perturbative-QCD scaling of Q²F(Q²) at large spacelike Q² |
| `positivity` | 50 | Im F ≥ 0 on the cut |
| `monotonicity` | 30 | phase non-decreasing along the cut |
| `mixing` | 1 | Gaussian prior on the ρ–ω mixing parameters |

## Loss-term ablations

`ablation-<term>.pt` is trained like `final.pt` with the weight of one term set to zero (`python main.py --ablation <term>`). One checkpoint exists for each of the eleven terms above:

`analyticity`, `asymptotics`, `dispersion`, `mixing`, `moments`, `monotonicity`, `positivity`, `pqcd`, `spacelike_data`, `timelike_data`, `watson`.

## Timelike data selections

These differ from `final.pt` in the timelike data they are trained on. The selection is made at the top of `TimelikeDataset` in `src/mesonforms/dataset_loader.py` and the file is chosen in `main.py`.

| File | Timelike data |
| --- | --- |
| `ablation-CMD3.pt` | e⁺e⁻ data including CMD-3 |
| `ablation-alltimelike.pt` | all e⁺e⁻ data (including CMD-3) and the τ data |
| `ablation-tautau.pt` | τ data only, uncorrected (`uncorrected_timelike_dataset.csv`) |
| `uncorrected-tautau.pt` | τ data only, uncorrected (earlier run) |
| `corrected-tautau.pt` | τ data only, corrected (earlier run) |

## Loss-weight variations

One weight is rescaled and the others are kept at baseline:

- `variation-<term>.pt`: weight × 1.5 (`python main.py --variation <term> --factor 1.5`),
- `variation--<term>.pt`: weight × 0.5 (`python main.py --variation <term>`).

Both exist for `analyticity`, `asymptotics`, `dispersion`, `mixing`, `moments`, `monotonicity`, `positivity`, `pqcd` and `watson`; the × 1.5 variation also for `spacelike_data` and `timelike_data`.

## Latin-hypercube sweep

`lhs/lhs-<i>.pt`, i = 0 … 63, vary all eleven weights at once. Each weight is drawn log-uniformly between half and twice its baseline in a 64-point Latin hypercube (seed 0). `python lhs_sweep.py` (in `scripts/`) prints the weights of every trial and `python lhs_sweep.py --train` retrains them, by default starting each trial from `final.pt`.

## Bootstrap ensemble

The 400 bootstrap fits behind the quoted uncertainties are not stored in the repository. `batch.sh` regenerates them into `ensemble/` (see the top-level [README](../README.md)).

## Earlier iterations

Checkpoints from earlier stages of the project, kept for the record. They predate the final data set and loss definitions and no script uses them.

| File | Committed | Origin |
| --- | --- | --- |
| `all-losses.pt` | 2026-05-20, updated until 2026-07-31 | all loss terms |
| `all-losses-old.pt` | 2026-05-27 | all loss terms, older training |
| `all-losses-moment.pt`, `all-losses-moment-alpha.pt` | 2026-05-27 | with the higher-moment sum rules added |
| `charge-radius.pt`, `watson.pt`, `analyticity.pt` | 2026-05-30 | first ablation series |
| `monotonicity.pt`, `spacelike.pt`, `timelike.pt` | 2026-06-01 | first ablation series |
| `moments.pt` | 2026-06-05 | first ablation series |
| `asymptotic.pt`, `pqcd.pt` | 2026-06-06 | first ablation series |
| `positivity.pt`, `dispersion.pt`, `mixing.pt` | 2026-06-23 to 2026-06-25 | first ablation series |
| `all-losses-corrected.pt` | 2026-08-04 | retrained on the corrected data |
| `all-losses-isovector.pt` | 2026-08-04 | pure isovector fit |
| `all-losses-minus-cmd3.pt` | 2026-08-04 | CMD-3 excluded |
| `all-losses-e+e-minus-CMD3.pt` | 2026-08-07 | e⁺e⁻ data without CMD-3 |
| `all-losses-roy.pt` | 2026-08-07 | Watson loss on a continuous Roy-equation phase |
| `roy.pt` | 2026-08-13 | after the update of the BELLE and CLEO data |
