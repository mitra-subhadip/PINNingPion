import math
import torch
import torch.nn as nn
from .constants import FOUR_M_PI2


class AdaptiveSine(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.a = nn.Parameter(torch.ones(in_features)) # Learnable frequency scaler

    def forward(self, x):
        return torch.sin(self.a * x)


class PINN_Fpi(nn.Module):
    def __init__(self, hidden_units=128, hidden_layers=4):
        super().__init__()

        # q^2 = x+iy; x and y are the inputs
        self.input_layer = nn.Linear(2, hidden_units)  

        self.hidden_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_units, hidden_units),
                AdaptiveSine(hidden_units),
            )
            for _ in range(hidden_layers)
        ])
        
        # F(q^2) = u+iv; u and v are the outputs
        self.output_layer = nn.Linear(hidden_units, 2)

        # Phase of the rho-omega mixing parameter in the multiplicative form F_I1 * (1 + eps * BW_omega).
        # Nearly real (~0 rad): the ~100 degree Orsay phase is already carried by arg F_I1(m_omega^2).
        self.phi_omega = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor([0.002], dtype=torch.float32))
        
        # Fixed physical constants for omega (in GeV)
        self.m_omega = 0.78266
        self.Gamma_omega = 0.00868

    def _forward_complex(self, z):
        # Extract x (Re) and y (Im) from z
        x, y = z[:, 0:1], z[:, 1:2]
        
        # Physically, for q^2 >= 4m_pi^2 on the real axis, we must evaluate slightly
        # in the upper half-plane (z + i*eps) to land on the correct physical sheet.
        eps = 1e-7
        y_safe = y + eps * (x >= FOUR_M_PI2).float()
        
        # Convert to PyTorch complex tensor
        z_complex = torch.complex(x, y_safe)
        
        # CONFORMAL MAPPING LAYER
        # Map z -> w using: w(z) = (sqrt(4m_pi^2 - z) - 2m_pi) / (sqrt(4m_pi^2 - z) + 2m_pi)
        sqrt_term = torch.sqrt(FOUR_M_PI2 - z_complex)
        sqrt_tc = math.sqrt(FOUR_M_PI2)
        w = (sqrt_term - sqrt_tc) / (sqrt_term + sqrt_tc)
        
        # Extract the real and imaginary parts of w to feed the standard real-valued MLP
        w_x = w.real.to(torch.float32)
        w_y = w.imag.to(torch.float32)
        mapped_z = torch.cat([w_x, w_y], dim=1)
        
        # Inference loop
        h = torch.sin(self.input_layer(mapped_z))
        for block in self.hidden_blocks:
            h = h + block(h)
        nn_out = self.output_layer(h)
        
        # Extract real, imaginary part of the neural network output
        u_nn, v_nn = nn_out[:, 0:1], nn_out[:, 1:2]

        # The pion form factor ansatz is: F(s) = 1 + s* NN(s).
        u_final = 1.0 + (w_x * u_nn - w_y * v_nn)
        v_final =       (w_x * v_nn + w_y * u_nn)
        return torch.cat([u_final, v_final], dim=1)
 
 
    def forward(self, q2_re, q2_im=None):
        """
        This function like `forward` evaluates the complex form factor just takes the real, imaginary
        input separately. In addition it enforces the Schwarz reflection constraint.
        """
        if q2_im is None:
            eps = 1e-7
            q2_im = torch.full_like(q2_re, eps)
            
        # Standardize shape to [N, 1] if they are 1D arrays
        if q2_re.dim() == 1:
            q2_re = q2_re.unsqueeze(1)
        if q2_im.dim() == 1:
            q2_im = q2_im.unsqueeze(1)
            
        # Concatenate spatial coordinates for the MLP input [N, 2]
        q2_im_eval = torch.abs(q2_im)  # Only evaluate in the upper half
        inputs = torch.cat([q2_re, q2_im_eval], dim=1)
        
        # Assuming your forward pass returns u, v
        out = self._forward_complex(inputs)
        u = out[:, 0:1]
        v = out[:, 1:2]
        
        u_final = u
        v_final = v * torch.sign(q2_im)  # As per Schwarz reflection principle
        return u_final.squeeze(-1), v_final.squeeze(-1) 


    def forward_w(self, w_x, w_y):
        """Bypasses the q^2 -> w mapping to evaluate directly in the conformal disk."""
        # Ensure correct shapes
        if w_x.dim() == 1:
            w_x = w_x.unsqueeze(1)
        if w_y.dim() == 1:
            w_y = w_y.unsqueeze(1)
            
        mapped_z = torch.cat([w_x, w_y], dim=1)
        
        # Pass through the network exactly as in your normal forward pass
        h = torch.sin(self.input_layer(mapped_z))
        for block in self.hidden_blocks:
            h = h + block(h)
        nn_out = self.output_layer(h)
        
        u_nn, v_nn = nn_out[:, 0:1], nn_out[:, 1:2]

        # Apply your exact same analytic continuation and normalization logic
        u_final = 1.0 + (w_x * u_nn - w_y * v_nn)
        v_final =       (w_x * v_nn + w_y * u_nn)
        return u_final.squeeze(-1), v_final.squeeze(-1)


    def forward_mixed(self, q2_re, omega_mask):
        """
        Evaluates the form factor for experimental data.
        If omega_mask=0, returns pure I=1 state (tau data). If omega_mask=1, adds omega
        interference (e+e- data). Note this is 1 - I for the dataset's `I` column.
        """
        q2_flat = q2_re.reshape(-1)
        mask_flat = omega_mask.reshape(-1)
        eps_tensor = torch.full_like(q2_flat, 1e-7)

        u_I1, v_I1 = self.forward(q2_flat, eps_tensor)
        F_I1 = torch.complex(u_I1, v_I1)
        F_full = F_I1
        
        # Calculate omega interference Breit-Wigner
        denom_real = (self.m_omega**2) - q2_flat
        denom_imag = - (self.m_omega * self.Gamma_omega) * torch.ones_like(q2_flat)
        denominator = torch.complex(denom_real, denom_imag)
        
        # Use m_omega^2 in numerator for proper dimensionless scaling
        BW_omega = (self.m_omega**2) / denominator
        
        # Apply omega mask
        mixing_term = self.alpha * torch.exp(1j * self.phi_omega) * BW_omega
        mask_complex = mask_flat.to(torch.complex64)
        F_full = F_I1 * (1.0 + mask_complex * mixing_term)
        return F_full.real, F_full.imag
