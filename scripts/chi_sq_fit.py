import numpy as np
import pandas as pd
import torch

from helper import load_model, set_torch_device

def calculate_global_chi2(model, df_space, df_time, device):
    """
    Calculates the exact Chi-Squared and Reduced Chi-Squared (χ²/N) 
    for both the spacelike and timelike experimental datasets.
    """
    model.eval()
    
    chi2_space = 0.0
    n_space = 0
    
    # Spacelike Region
    if df_space is not None:
        # Note the minus sign: dataset has Q^2, network needs q^2 = -Q^2
        q2 = torch.tensor(-df_space['q2'].values, dtype=torch.float32, device=device)
        
        # Target values and propagated errors
        F_true_sq = df_space['Fpi_sq'].values
        err_sq = df_space['error'].values
        
        # Reconstruct the linear form factor and linear error
        F_true_np = np.sqrt(F_true_sq)
        err_linear_np = err_sq / (2.0 * F_true_np + 1e-8)
        
        F_true = torch.tensor(F_true_np, dtype=torch.float32, device=device)
        err_linear = torch.tensor(err_linear_np, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            u_pred, _ = model.forward(q2, torch.zeros_like(q2))
            u_pred = u_pred.squeeze()
            
            # Linear Chi-Squared calculation
            residuals = ((u_pred - F_true) / err_linear)**2
            chi2_space = torch.sum(residuals).item()
            n_space = len(q2)
                

    # Timelike Region
    chi2_time = 0.0
    n_time = 0
    
    if df_time is not None:
        s = torch.tensor(df_time['q2'].values, dtype=torch.float32, device=device)
        
        # The crucial Isospin flag (I=1 for tau, I=0 for e+e-)
        I_val = torch.tensor(df_time['I'].values, dtype=torch.float32, device=device)
        
        # Target values and raw experimental errors
        F_sq_true = torch.tensor(df_time['Fpi_sq'].values, dtype=torch.float32, device=device)
        err_sq = torch.tensor(df_time['error'].values, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Use forward_mixed to correctly apply rho-omega mixing
            u_pred, v_pred = model.forward_mixed(s, 1 - I_val)
            u_pred = u_pred.squeeze()
            v_pred = v_pred.squeeze()
            
            F_sq_pred = u_pred**2 + v_pred**2
            
            # Squared Chi-Squared calculation
            residuals = ((F_sq_pred - F_sq_true) / err_sq)**2
            chi2_time = torch.sum(residuals).item()
            n_time = len(s)
                
    print("GLOBAL CHI-SQUARED")
    
    if n_space > 0:
        red_chi2_space = chi2_space / n_space
        print(f"Spacelike Data : χ² = {chi2_space:7.2f} | N = {n_space:<4} | χ²/N = {red_chi2_space:.3f}")
        
    if n_time > 0:
        red_chi2_time = chi2_time / n_time
        print(f"Timelike Data  : χ² = {chi2_time:7.2f} | N = {n_time:<4} | χ²/N = {red_chi2_time:.3f}")
        
    if n_space > 0 and n_time > 0:
        total_chi2 = chi2_space + chi2_time
        total_n = n_space + n_time
        print(f"TOTAL          : χ² = {total_chi2:7.2f} | N = {total_n:<4} | χ²/N = {total_chi2/total_n:.3f}")
    
    return chi2_space, chi2_time

if __name__ == "__main__":
    df_spacelike_data = pd.read_csv("../dataset/spacelike_dataset.csv")
    df_timelike_data = pd.read_csv("../dataset/timelike_dataset.csv")
    
    DEVICE = set_torch_device()
    model = load_model(DEVICE)
    chi2_space, chi2_time = calculate_global_chi2(model, df_spacelike_data, df_timelike_data, device=DEVICE)
