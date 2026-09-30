from helper import load_model, set_torch_device

if __name__ == "__main__":
    DEVICE = set_torch_device()
    model = load_model(DEVICE)
    print(f"Alpha: {model.alpha}")
    print(f"Phi_omega: {model.phi_omega}")
