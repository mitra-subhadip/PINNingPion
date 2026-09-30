import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mesonforms.dataset_loader import SpacelikeDataset, TimelikeDataset
from mesonforms.helper import experimental_delta11, get_experimental_phase_interpolator
from mesonforms.model import PINN_Fpi
from mesonforms.train import PINNTrainer

WEIGHTS = {
    'analyticity': 400.0, 'watson': 300.0, 'monotonicity': 30,
    'timelike_data': 1000.0, 'spacelike_data': 800.0, 'moments': 400.0, 'asymptotics': 300.0,
    'pqcd': 50.0, 'positivity': 50.0, 'dispersion': 100.0, 'mixing': 1,
}

# Parameters
BATCH_SIZE = 128
LR = 5e-5
N_EPOCHS = 3000
BOOTSTRAP = False
PT_FILE = "weights/final.pt"

parser = argparse.ArgumentParser(description="Train the PINN for the pion form factor.")
parser.add_argument("--ablation", choices=WEIGHTS, help="Loss term to switch off")
parser.add_argument("--variation", choices=WEIGHTS, help="Loss term whose weight is multiplied by --factor")
parser.add_argument("--factor", type=float, default=0.5, help="Factor applied to the --variation weight")
parser.add_argument("--seed", type=int, default=0, help="Seed for bootstrapping the dataset")
args = parser.parse_args()

if args.variation:
    WEIGHTS[args.variation] *= args.factor
    PT_FILE = f"weights/variation-{'-' if args.factor < 1 else ''}{args.variation}.pt"

if args.ablation:
    WEIGHTS[args.ablation] = 0.0
    PT_FILE = f"weights/ablation-{args.ablation}.pt"

# The seed is required for the error analysis (bootstrapping the dataset)
if args.seed:
    BOOTSTRAP = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    PT_FILE = f"ensemble/error-{args.ablation or 'no-ablation'}-{args.seed}.pt"

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = 'cpu'

    # Load the dataset
    df_spacelike_data = pd.read_csv("dataset/spacelike_dataset.csv")
    spacelike_dataset = SpacelikeDataset(df_spacelike_data, device=DEVICE, bootstrap=BOOTSTRAP)

    df_timelike_data = pd.read_csv("dataset/timelike_dataset.csv")
    timelike_dataset = TimelikeDataset(df_timelike_data, device=DEVICE, bootstrap=BOOTSTRAP)

    spacelike_loader = DataLoader(spacelike_dataset, batch_size=BATCH_SIZE, shuffle=True)
    timelike_loader = DataLoader(timelike_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the interpolator once
    phase_data = pd.read_csv("dataset/colangelo_phase.csv")
    phase_interpolator = get_experimental_phase_interpolator(phase_data['q'].values**2, phase_data['phase'].values)

    Path(PT_FILE).parent.mkdir(exist_ok=True)

    # Train in rounds, resuming from the checkpoint, until the loss has converged
    while True:
        model = PINN_Fpi(hidden_units=128, hidden_layers=4).to(DEVICE)
        trainer = PINNTrainer(model, loss_weights=WEIGHTS, phase_fn=experimental_delta11)
        trainer.load_checkpoint(PT_FILE)

        print(f"Starting PINN training ({N_EPOCHS}, {LR})")
        loss = trainer.train(
            n_epochs=N_EPOCHS,
            lr=LR,
            continue_training=1,
            spacelike_loader=spacelike_loader,
            timelike_loader=timelike_loader,
            phase_interpolator=phase_interpolator,
            device=DEVICE,
            verbose=True,
        )
        trainer.save_checkpoint(PT_FILE)

        if loss >= 2500.0:
            LR = 1e-3
            N_EPOCHS = 5000
        elif loss >= 1200:
            LR = 5e-4
            N_EPOCHS = 4000
        elif loss >= 80:
            LR = 1e-4
            N_EPOCHS = 3000
        elif loss >= 32:
            LR = 5e-5
            N_EPOCHS = 2000
        else:
            break
