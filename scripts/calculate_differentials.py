import torch

from helper import load_model, set_torch_device
from mesonforms.constants import CONV_FM2_GEV2


def extract_pion_moments(model, device):
    """
    Extracts the pion charge radius <r^2_pi> and the curvature <r^4_pi> 
    using the exact Taylor expansion of the PINN at q^2 = 0.
    """
    q2_0 = torch.tensor([0.0], device=device, requires_grad=True)
    q2_im_0 = torch.tensor([0.0], device=device)
    
    # Evaluate the network's real part (Re(Fpi))
    u_0, _ = model.forward(q2_0, q2_im_0)
    
    # Compute the first derivative F'(0) w.r.t q^2
    F_prime = torch.autograd.grad(outputs=u_0.sum(), inputs=q2_0, create_graph=True)[0]
    f_prime_0 = F_prime.item()
    
    # Compute the second derivative F''(0) w.r.t q^2
    F_double_prime = torch.autograd.grad(outputs=F_prime.sum(), inputs=q2_0, create_graph=True)[0]
    f_double_prime_0 = F_double_prime.item()
    
    # Compute the third derivative F'''(0) w.r.t q^2
    F_triple_prime = torch.autograd.grad(outputs=F_double_prime.sum(), inputs=q2_0)[0]
    f_triple_prime_0 = F_triple_prime.item()
    
    # Calculate Moments based on the Taylor expansion:
    r2_pi_gev2 = 6.0 * f_prime_0
    r4_pi_gev4 = 60.0 * f_double_prime_0
    r6_pi_gev6 = 840.0 * f_triple_prime_0
    
    # Convert from GeV units to fm units
    r2_pi_fm2 = r2_pi_gev2 / CONV_FM2_GEV2
    r4_pi_fm4 = r4_pi_gev4 / (CONV_FM2_GEV2 ** 2)
    r6_pi_fm6 = r6_pi_gev6 / (CONV_FM2_GEV2 ** 3)
    return r2_pi_fm2, r4_pi_fm4, r6_pi_fm6


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    r2_extracted, r4_extracted, r6_extracted = extract_pion_moments(model, DEVICE)
    print("--- Extracted Pion Moments ---")
    print(f"Charge Radius <r^2_pi> = {r2_extracted:.6f} fm^2")
    print(f"Curvature     <r^4_pi> = {r4_extracted:.6f} fm^4")
    print(f"6th Moment    <r^6_pi> = {r6_extracted:.6f} fm^6")
