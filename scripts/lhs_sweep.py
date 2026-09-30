"""Latin Hypercube Sampling (LHS) design for the MesonForms PINN hyperparameters.

Draws N points from the unit hypercube with ``scipy.stats.qmc.LatinHypercube``
(scrambled, centred-discrepancy optimised, fixed seed) and maps every point to a
concrete hyperparameter configuration through :func:`unit_to_params`.

The search space is a plain dictionary. Each entry is a *spec* dict whose
``"type"`` key selects how a unit-interval coordinate ``u in [0, 1)`` is mapped:

    uniform      {"type": "uniform",     "low": a, "high": b}          -> a + u (b - a)
    loguniform   {"type": "loguniform",  "low": a, "high": b}   (a>0)  -> exp(log a + u (log b - log a))
    int          {"type": "int",         "low": a, "high": b}          -> integer in [a, b], inclusive, uniform
    logint       {"type": "logint",      "low": a, "high": b}   (a>0)  -> integer in [a, b], log-spaced
    categorical  {"type": "categorical", "choices": [...]}             -> one choice, equal mass per choice
    bool         {"type": "bool"}                                      -> u < 0.5
    fixed        {"type": "fixed",       "value": v}                   -> v  (no LHS dimension consumed)

Every spec except ``fixed`` consumes exactly one LHS dimension, so the design has
``len(space) - #fixed`` dimensions.

Usage
-----
    python lhs_sweep.py                 # N=64, seed=0, physics weights only
    python lhs_sweep.py -n 80 --seed 7
    python lhs_sweep.py --include-arch  # also sample architecture / optimiser knobs

    # Train the sweep (dataset/ and weights/ paths are relative to the repo root):
    python lhs_sweep.py --train                        # all 64 trials -> weights/lhs/lhs-<trial>.pt
    python lhs_sweep.py --train --trials 0-15          # a slice, e.g. one machine of several
    python lhs_sweep.py --train --trials 3,7,42        # specific trials
    Existing checkpoints are skipped, so the command can be re-run to resume.

    from lhs_sweep import lhs_design, build_space
    configs = lhs_design(build_space(), n=64, seed=0)   # list[dict], one per trial
    for cfg in configs:
        WEIGHTS = {k: cfg[k] for k in BASELINE_WEIGHTS}
        ...
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

# ----------------------------------------------------------------------------
# Search space
# ----------------------------------------------------------------------------

# Baseline loss weights from main.py. Each weight is scanned log-uniformly over
# [baseline / SPAN, baseline * SPAN] so that a factor-of-k change up or down is
# equally likely, which is the natural prior for multiplicative loss weights.
BASELINE_WEIGHTS: dict[str, float] = {
    "analyticity": 400.0,
    "watson": 300.0,
    "monotonicity": 30.0,
    "timelike_data": 1000.0,
    "spacelike_data": 800.0,
    "moments": 400.0,
    "asymptotics": 300.0,
    "pqcd": 50.0,
    "positivity": 50.0,
    "dispersion": 100.0,
    "mixing": 1.0,
}
SPAN = 2.0


def _log_around(center: float, span: float = SPAN) -> dict[str, Any]:
    return {"type": "loguniform", "low": center / span, "high": center * span}


# Physics loss weights: 11 log-uniform dimensions.
PHYSICS_SPACE: dict[str, dict[str, Any]] = {
    name: _log_around(w) for name, w in BASELINE_WEIGHTS.items()
}

# Optional architecture / optimiser knobs; these exercise the remaining spec types.
ARCH_SPACE: dict[str, dict[str, Any]] = {
    "hidden_units": {"type": "categorical", "choices": [64, 128, 256]},
    "hidden_layers": {"type": "int", "low": 3, "high": 6},
    "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
    "batch_size": {"type": "logint", "low": 32, "high": 256},
    "bootstrap": {"type": "bool"},
    "n_epochs": {"type": "fixed", "value": 3000},
}


def build_space(include_arch: bool = False) -> dict[str, dict[str, Any]]:
    space = dict(PHYSICS_SPACE)
    if include_arch:
        space.update(ARCH_SPACE)
    return space


# ----------------------------------------------------------------------------
# Unit hypercube -> hyperparameters
# ----------------------------------------------------------------------------

def sampled_names(space: dict[str, dict[str, Any]]) -> list[str]:
    """Names of the entries that consume an LHS dimension, in dictionary order."""
    return [k for k, spec in space.items() if spec["type"] != "fixed"]


def unit_to_params(u: np.ndarray, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Map one point ``u`` of the unit hypercube to real hyperparameter values.

    ``u`` has one coordinate per non-fixed entry of ``space`` (dictionary order),
    each in [0, 1). The discrete branches clip the top edge so u == 1.0 is safe too.
    """
    names = sampled_names(space)
    if len(u) != len(names):
        raise ValueError(f"u has {len(u)} coordinates, space has {len(names)} sampled dims")

    params: dict[str, Any] = {}
    it = iter(u)
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "fixed":
            params[name] = spec["value"]
            continue

        x = float(next(it))
        if kind == "uniform":
            lo, hi = spec["low"], spec["high"]
            params[name] = lo + x * (hi - lo)
        elif kind == "loguniform":
            lo, hi = spec["low"], spec["high"]
            if lo <= 0:
                raise ValueError(f"{name}: loguniform requires low > 0, got {lo}")
            params[name] = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))
        elif kind == "int":
            lo, hi = int(spec["low"]), int(spec["high"])
            params[name] = min(lo + int(math.floor(x * (hi - lo + 1))), hi)
        elif kind == "logint":
            lo, hi = int(spec["low"]), int(spec["high"])
            if lo <= 0:
                raise ValueError(f"{name}: logint requires low > 0, got {lo}")
            # Log-space over [lo, hi + 1) so every integer gets non-zero mass.
            val = math.exp(math.log(lo) + x * (math.log(hi + 1) - math.log(lo)))
            params[name] = min(int(math.floor(val)), hi)
        elif kind == "categorical":
            choices = spec["choices"]
            params[name] = choices[min(int(math.floor(x * len(choices))), len(choices) - 1)]
        elif kind == "bool":
            params[name] = bool(x < 0.5)
        else:
            raise ValueError(f"{name}: unknown spec type {kind!r}")
    return params


# ----------------------------------------------------------------------------
# Design generation
# ----------------------------------------------------------------------------

def lhs_unit_sample(d: int, n: int, seed: int) -> np.ndarray:
    """Scrambled, discrepancy-optimised LHS in [0, 1)^d. Shape (n, d)."""
    sampler = qmc.LatinHypercube(d=d, scramble=True, optimization="random-cd", seed=seed)
    return sampler.random(n=n)


def lhs_design(space: dict[str, dict[str, Any]], n: int = 64, seed: int = 0) -> list[dict[str, Any]]:
    """Return ``n`` hyperparameter configurations, one dict per trial."""
    unit = lhs_unit_sample(len(sampled_names(space)), n, seed)
    return [unit_to_params(u, space) for u in unit]


# ----------------------------------------------------------------------------
# Training (mirrors main.py)
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# main.py's adaptive schedule: (loss threshold, next lr, next n_epochs). Below the last
# threshold the run is accepted and saved. Thresholds are tuned for the baseline weight
# sum, so they are rescaled by sum(w) / sum(w_baseline) unless --no-scale-thresholds.
SCHEDULE = [(2500.0, 1e-3, 5000), (1200.0, 5e-4, 4000), (80.0, 1e-4, 3000), (31.0, 5e-5, 2000)]


def parse_trials(spec: str, n: int) -> list[int]:
    """'0-15' -> [0..15], '3,7,42' -> [3,7,42], '' -> all."""
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [t for t in out if not 0 <= t < n]
    if bad:
        raise SystemExit(f"trials {bad} outside 0..{n - 1}")
    return out


def train_trial(weights: dict[str, float], out_file: Path, loaders, phase_interpolator, device, *,
                init_ckpt: Path | None, lr: float, n_epochs: int, max_rounds: int,
                scale_thresholds: bool, seed: int, verbose: bool) -> float:
    """Train one configuration with main.py's loop and save the best checkpoint. Returns final loss."""
    import torch
    from mesonforms.helper import experimental_delta11
    from mesonforms.model import PINN_Fpi
    from mesonforms.train import PINNTrainer

    spacelike_loader, timelike_loader = loaders
    torch.manual_seed(seed); np.random.seed(seed)
    scale = sum(weights[k] for k in BASELINE_WEIGHTS) / sum(BASELINE_WEIGHTS.values()) if scale_thresholds else 1.0

    model = PINN_Fpi(hidden_units=128, hidden_layers=4).to(device)
    trainer = PINNTrainer(model, loss_weights=weights, phase_fn=experimental_delta11)
    if init_ckpt is not None:
        trainer.load_checkpoint(str(init_ckpt))

    loss = float("inf")
    for rnd in range(max_rounds):
        print(f"  round {rnd + 1}/{max_rounds}: {n_epochs} epochs at lr={lr:g}")
        loss = trainer.train(n_epochs=n_epochs, lr=lr, continue_training=1,
                             spacelike_loader=spacelike_loader, timelike_loader=timelike_loader,
                             phase_interpolator=phase_interpolator, device=device, verbose=verbose)
        trainer.save_checkpoint(str(out_file))
        for thr, next_lr, next_epochs in SCHEDULE:
            if loss >= thr * scale:
                lr, n_epochs = next_lr, next_epochs
                break
        else:
            break  # below every threshold: converged
    return loss


def run_sweep(args) -> None:
    import pandas as pd
    from torch.utils.data import DataLoader
    from helper import set_torch_device
    from mesonforms.dataset_loader import SpacelikeDataset, TimelikeDataset
    from mesonforms.helper import get_experimental_phase_interpolator

    space = build_space(args.include_arch)
    configs = lhs_design(space, args.n_samples, args.seed)
    trials = parse_trials(args.trials, args.n_samples)
    out_dir = (ROOT / args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = (ROOT / args.init).resolve() if args.init else None
    if init_ckpt is not None and not init_ckpt.exists():
        raise SystemExit(f"initial checkpoint not found: {init_ckpt}")

    device = set_torch_device()
    print(f"device={device}  trials={trials[0]}..{trials[-1]} ({len(trials)})  out={out_dir}  init={init_ckpt}")

    df_sp = pd.read_csv(ROOT / args.spacelike); df_tl = pd.read_csv(ROOT / args.timelike)
    spacelike_loader = DataLoader(SpacelikeDataset(df_sp, device=device), batch_size=args.batch_size, shuffle=True)
    timelike_loader = DataLoader(TimelikeDataset(df_tl, device=device), batch_size=args.batch_size, shuffle=True)
    phase = pd.read_csv(ROOT / "dataset/colangelo_phase.csv")
    phase_interpolator = get_experimental_phase_interpolator(phase["q"].values ** 2, phase["phase"].values)

    for t in trials:
        out_file = out_dir / f"lhs-{t}.pt"
        if out_file.exists() and not args.overwrite:
            print(f"trial {t}: {out_file.name} exists, skipping"); continue
        weights = {k: float(configs[t][k]) for k in BASELINE_WEIGHTS}
        print(f"\ntrial {t}: " + ", ".join(f"{k}={v:.4g}" for k, v in weights.items()))
        t0 = time.time()
        loss = train_trial(weights, out_file, (spacelike_loader, timelike_loader), phase_interpolator, device,
                           init_ckpt=init_ckpt, lr=args.lr, n_epochs=args.epochs, max_rounds=args.max_rounds,
                           scale_thresholds=not args.no_scale_thresholds, seed=args.seed * 10_000 + t,
                           verbose=args.verbose)
        print(f"trial {t}: final loss {loss:.3f}  ({(time.time() - t0) / 60:.1f} min)  -> {out_file}")


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an LHS sweep over MesonForms hyperparameters.")
    ap.add_argument("-n", "--n-samples", type=int, default=64, help="number of LHS points (50-100 typical)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the scrambled LHS")
    ap.add_argument("--include-arch", action="store_true", help="also sample the architecture/optimiser block")
    tr = ap.add_argument_group("training (--train)")
    tr.add_argument("--train", action="store_true", help="train the design instead of only printing it")
    tr.add_argument("--trials", default="", help="which trials: '0-15', '3,7,42' (default: all)")
    tr.add_argument("--out-dir", default="weights/lhs", help="checkpoint directory, relative to repo root")
    tr.add_argument("--init", default="weights/final.pt", help="checkpoint every trial starts from ('' = scratch)")
    tr.add_argument("--overwrite", action="store_true", help="retrain trials whose checkpoint exists")
    tr.add_argument("--spacelike", default="dataset/spacelike_dataset.csv")
    tr.add_argument("--timelike", default="dataset/timelike_dataset.csv")
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--lr", type=float, default=5e-5, help="learning rate of the first round")
    tr.add_argument("--epochs", type=int, default=3000, help="epochs of the first round")
    tr.add_argument("--max-rounds", type=int, default=6, help="cap on adaptive rounds per trial (main.py loops forever)")
    tr.add_argument("--no-scale-thresholds", action="store_true", help="use main.py's absolute loss thresholds unscaled")
    tr.add_argument("--verbose", action="store_true", help="per-100-epoch loss breakdown")
    args = ap.parse_args()

    if args.train:
        run_sweep(args)
        return

    space = build_space(args.include_arch)
    names = sampled_names(space)
    unit = lhs_unit_sample(len(names), args.n_samples, args.seed)
    configs = [unit_to_params(u, space) for u in unit]

    print(f"LHS design: n={args.n_samples}, d={len(names)}, seed={args.seed}")
    print(f"sampled dims : {names}")
    print(f"fixed dims   : {[k for k in space if k not in names]}")
    print(f"centred L2 discrepancy of unit sample: {qmc.discrepancy(unit, method='CD'):.5f}\n")

    cols = list(configs[0].keys())
    width = {c: max(len(c), *(len(_fmt(cfg[c])) for cfg in configs)) for c in cols}
    print("trial  " + "  ".join(c.rjust(width[c]) for c in cols))
    for i, cfg in enumerate(configs):
        print(f"{i:5d}  " + "  ".join(_fmt(cfg[c]).rjust(width[c]) for c in cols))


if __name__ == "__main__":
    main()

