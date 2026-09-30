# PINNing Pion

A physics-informed neural network (PINN) for the electromagnetic form factor of the pion, F<sub>π</sub>(q²), over the whole complex q² plane.

The network takes a complex q² and returns Re F and Im F. The input is first mapped onto the unit disk, w = (√(4m<sub>π</sub>² − q²) − 2m<sub>π</sub>) / (√(4m<sub>π</sub>² − q²) + 2m<sub>π</sub>), which puts the two-pion cut on the unit circle, and the output is built as F = 1 + w·NN(w), so that F(0) = 1 holds exactly. It is trained on the spacelike and timelike data together with the constraints a form factor has to satisfy: analyticity, Watson's theorem, a dispersion relation and its sum rules, positivity of Im F and a monotonic phase along the cut, and the asymptotic behaviour (F → 0 at infinity, perturbative-QCD scaling at large spacelike momenta). For the e⁺e⁻ data the isovector form factor is multiplied by a ρ–ω mixing term with two fitted parameters.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/mesonforms/` | the package: network (`model.py`), loss terms and training loop (`train.py`), data weighting (`dataset_loader.py`) |
| `main.py` | trains a fit the model |
| `dataset/` | experimental data and the ππ phase shifts, with their sources ([README](dataset/README.md)) |
| `weights/` | trained checkpoints ([README](weights/README.md)) |
| `scripts/` | analysis and plotting scripts ([README](scripts/README.md)) |

## Installation

Clone the project,

```bash
git clone https://github.com/mitra-subhadip/PINNingPion.git
cd PINNingPion
```

or download it as a zip from GitHub (under the green Code button).

Python 3.11 or newer is required. Installing the pinned dependencies from `pylock.toml` needs pip ≥ 26; upgrade it if necessary and then install the dependencies and the package:

```bash
python -m pip install --upgrade pip
pip install -r pylock.toml
pip install .
```

## Using the trained network

The analysis scripts evaluate the published fit, `weights/final.pt`, and are run from the `scripts` directory:

```bash
cd scripts
python <script-name>.py
```

[`scripts/README.md`](scripts/README.md) lists what each script produces, and [`weights/README.md`](weights/README.md) shows how to load a checkpoint in your own code.

## Training

```bash
python main.py
```

trains in rounds of a few thousand Adam epochs. Each round reloads the checkpoint `PT_FILE`, trains, and saves the best state back to it; the learning rate and length of the next round are chosen from the loss reached, and training stops once the loss is below 32.

`PT_FILE` is `weights/final.pt`, so the plain command continues the published fit and overwrites that file. To train from scratch, point `PT_FILE` at a file that does not exist yet. Batch size, learning rate, number of epochs and the loss weights are set at the top of `main.py`.

The other fits in `weights/` are variations of this one:

```bash
python main.py --ablation watson                  # one loss term switched off  -> weights/ablation-watson.pt
python main.py --variation watson                 # its weight halved           -> weights/variation--watson.pt
python main.py --variation watson --factor 1.5    # its weight times 1.5        -> weights/variation-watson.pt
python main.py --seed 7                           # bootstrap replica of the data -> ensemble/error-no-ablation-7.pt
```

The timelike data entering the fit (by default the e⁺e⁻ data without CMD-3) are selected at the top of `TimelikeDataset` in `src/mesonforms/dataset_loader.py`.

### Bootstrap ensemble

The uncertainties are obtained from 500 fits, each trained from scratch on data resampled within their errors. They are not stored in the repository, separately request us in case you require those ensemble weights.

## License

Released under the MIT License, see [LICENSE](LICENSE).
