import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


__all__ = ['SpacelikeDataset', 'TimelikeDataset']


class SpacelikeDataset(Dataset):
    def __init__(self, df, device, n_bins=20, bootstrap=False):

        # Actual F_pi_sq, standard deviation
        F = np.sqrt(df['Fpi_sq'].values)
        sigma_F2 = df['error'].values
        
        # Sample based on gaussian distribution
        if bootstrap:
            F = np.random.normal(F, sigma_F2)

        # Propagate experimental error: sigma_F = sigma_F2 / (2 * F)
        sigma_F = sigma_F2 / (2.0 * F + 1e-8)
        
        # Calculate Relative Statistical Weight: (F / sigma_F)^2
        # This flattens the dynamic range perfectly.
        relative_weight = (F / (sigma_F + 1e-8))**2
        
        # Density weighting into bins
        df['bin'] = pd.cut(df['q2'], bins=n_bins, include_lowest=True)
        bin_counts = df['bin'].value_counts()
        dens_weight = df['bin'].map(lambda x: 1.0 / bin_counts[x] if bin_counts[x] > 0 else 0.0).values
        combined_weight = relative_weight * dens_weight
        
        # Normalize weights
        self.q2 = torch.tensor(df['q2'].values, dtype=torch.float32).to(device)
        self.weight = torch.tensor(combined_weight / combined_weight.mean(), dtype=torch.float32).to(device)
        self.F = torch.tensor(F, dtype=torch.float32).to(device)

    def __getitem__(self, idx):
        return -self.q2[idx], self.F[idx], self.weight[idx]

    def __len__(self):
        return len(self.q2)


class TimelikeDataset(Dataset):
    def __init__(self, df, device, n_bins=50, bootstrap=False):
        df = df[df["Source"] != "CMD-3"]
        df = df[df["I"] == 0]

        # I is for the channel value (I=1 for pure isovector, I=0 for isovector + isoscalar)
        self.I = torch.tensor(df['I'].values, dtype=torch.float32).to(device)
        self.q2 = torch.tensor(df['q2'].values, dtype=torch.float32).to(device)
        self.Fpi_sq = torch.tensor(df['Fpi_sq'].values, dtype=torch.float32).to(device)

        # Calculate Standard Statistical Weight (1/sigma^2)
        # We clamp the error to avoid division by zero
        error = torch.tensor(df['error'].values, dtype=torch.float32).to(device)
        stat_weight = 1.0 / (error + 1e-8)
        
        # Sample based on gaussian distribution
        if bootstrap:
            self.Fpi_sq = torch.normal(self.Fpi_sq, error)

        # We create 'n_bins' equal-width bins across the q^2 range
        df['bin'] = pd.cut(df['q2'], bins=n_bins, include_lowest=True)
        bin_counts = df['bin'].value_counts()
        df['density_weight'] = df['bin'].map(lambda x: 1.0 / bin_counts[x] if bin_counts[x] > 0 else 0.0)
        
        dens_weight = torch.tensor(df['density_weight'].values, dtype=torch.float32).to(device)
        combined_weight = stat_weight * dens_weight

        # Find which indices belong to I=0, I=1
        ee_mask = (self.I == 0.0)
        tau_mask = (self.I == 1.0)
        
        # Calculate the total weight of each dataset
        tau_sum = combined_weight[tau_mask].sum()
        ee_sum = combined_weight[ee_mask].sum()
        
        # Scale the tau weights so their total influence matches the e+e- data
        if tau_sum > 0 and ee_sum > 0:
            combined_weight[tau_mask] *= (ee_sum / tau_sum)
            
        # Normalize the final weights so the mean is 1.0 (keeps learning rates stable)
        self.weight = combined_weight / combined_weight.mean()

    def __getitem__(self, idx):
        return self.q2[idx], self.I[idx], self.Fpi_sq[idx], self.weight[idx]

    def __len__(self):
        return len(self.q2)
