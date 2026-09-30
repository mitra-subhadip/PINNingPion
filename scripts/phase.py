import matplotlib.pyplot as plt
import torch

from helper import load_model, phase_degrees, set_torch_device


if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)

    q2_grid = torch.linspace(0.08, 10.0, 5000, device=DEVICE)

    plt.figure(figsize=(8, 6))
    plt.plot(q2_grid.cpu().numpy(), phase_degrees(model, q2_grid))
    plt.xlabel(r'$q^2 \text{ (GeV}^2\text{)}$')
    plt.ylabel(r'$|arg(F_{\pi}(q^2)|^2)$')
    plt.title(r'$|arg(F_{\pi}(q^2)|^2)$')
    plt.savefig('Phase.png')
