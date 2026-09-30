import copy
import math
import os

import torch
from .constants import FOUR_M_PI2
from .helper import experimental_delta11, strong_coupling_2loop


class PINNTrainer:
    def __init__(self, model, loss_weights, phase_fn):
        self.model = model
        self.phase_fn = phase_fn
        self.loss_weights = loss_weights
        self.best_loss_val = 1e10

    def _loss_pqcd(self, device):
        q2_deep = torch.linspace(-1000.0, -5.0, 5000, device=device).requires_grad_(True)
        u_deep, _ = self.model.forward(q2_deep)
        
        Q_sq = torch.abs(q2_deep)
        
        # Calculate alpha_s(Q^2) using the PyTorch-safe function
        alpha_s = strong_coupling_2loop(Q_sq).to(device)
        
        # Brodsky-Lepage Scaling: F(Q^2) ~ alpha_s(Q^2) / Q^2
        # Therefore, (Q^2 * F(Q^2)) / (alpha_s * (1 + 0.184 * alpha_s)) should be like a constant at infinity
        scaled_f = (Q_sq * u_deep) / (alpha_s * (1 + 0.184 * alpha_s))
        
        # The derivative of this scaled function w.r.t q2 should be zero
        grad_scaled = torch.autograd.grad(
            outputs=scaled_f.sum(), 
            inputs=q2_deep, 
            create_graph=True
        )[0]
        return (grad_scaled**2).mean()


    def _loss_monotonicity_w(self, device):
        """
        Ensures the phase is monotonically increasing across the ENTIRE physical cut
        by evaluating the derivative directly along the conformal boundary.
        """
        # Sweep the cut from threshold (-pi) to infinity (0)
        theta = torch.linspace(-math.pi + 1e-4, -1e-4, 1000, device=device).requires_grad_(True)
        
        w_x = torch.cos(theta)
        w_y = torch.sin(theta)
        
        u, v = self.model.forward_w(w_x, w_y)
        
        # Calculate derivatives of u and v with respect to theta
        u_grad = torch.autograd.grad(u.sum(), theta, create_graph=True)[0]
        v_grad = torch.autograd.grad(v.sum(), theta, create_graph=True)[0]
        
        # Analytically calculate the derivative of the phase: d(delta)/d(theta)
        # d(delta)/d(theta) = (u * v' - v * u') / (u^2 + v^2)
        magnitude_sq = (u**2 + v**2) + 1e-8
        phase_grad = (u * v_grad - v * u_grad) / magnitude_sq
        
        # Penalize any negative phase gradients
        return (torch.relu(-phase_grad)**2).mean()


    def _loss_watson(self, q2_re, phase_interpolator, device):
        """
        Loss for Watson's theorem: Arg(F(q^2)) = delta_11(q^2).
        Strictly valid only in the elastic region: 4M_pi^2 < q^2 < ~1.05 GeV^2.
        """
        q2_re = q2_re.clone().detach().requires_grad_(True)
        
        # Only enforce Watson's theorem in the elastic region
        elastic_mask = (q2_re > FOUR_M_PI2) & (q2_re < 1.0)
        
        if not torch.any(elastic_mask):
            return torch.tensor(0.0, device=device)
            
        q2_elastic = q2_re[elastic_mask]
        
        # Evaluate on the upper half of the physical cut
        u_m, v_m = self.model.forward(q2_elastic)
        
        # Get target phase
        delta_m = experimental_delta11(q2_elastic, phase_interpolator, device)
        cos_delta = torch.cos(delta_m)
        sin_delta = torch.sin(delta_m)
        
        # Cross-Product Phase Residual (Weighted by Magnitude)
        # u*sin(delta) - v*cos(delta) = |F| * sin(delta - theta_pred)
        # This smoothly forces the angle to match without blowing up gradients when |F| is small.
        phase_residual = (u_m * sin_delta) - (v_m * cos_delta)
        
        # Soft direction penalty to prevent 180-degree flips
        # (u*cos + v*sin) = |F| * cos(theta_pred - delta)
        # We penalize negative cosine to ensure it points in the same hemisphere
        direction_cos = (u_m * cos_delta) + (v_m * sin_delta)
        direction_penalty = torch.relu(-direction_cos)
        
        return (phase_residual**2).mean() + (direction_penalty**2).mean()


    def _loss_positivity_w(self, device):
        """
        Enforces Im(F(w)) >= 0 strictly on the physical branch cut.
        By evaluating on the conformal boundary (theta from -pi to 0), 
        this guarantees positivity all the way to q^2 = infinity without 
        numerical truncation.
        """
        # Sample the physical branch cut in conformal angle [-pi, 0]
        theta = torch.linspace(-math.pi + 1e-4, -1e-4, 1000, device=device)
        
        w_cut_x = torch.cos(theta)
        w_cut_y = torch.sin(theta)
        
        _, v_boundary = self.model.forward_w(w_cut_x, w_cut_y)
        v_boundary = v_boundary.squeeze()
        
        # Penalize any negative imaginary parts
        # torch.relu(-v) is 0 if v >= 0, and strictly positive if v < 0
        return (torch.relu(-v_boundary)**2).mean()
    

    def _loss_asymptotics(self, device):
        """Enforces F(infinity) = 0, F'(infinity)=0 by evaluating exactly at w = 1 + 0i"""
        
        # Define the exact point at infinity in the conformal disk
        w_inf_x = torch.linspace(0.99, 1.0, 50, device=device).requires_grad_(True)
        w_inf_y = torch.zeros_like(w_inf_x, device=device).requires_grad_(True)
        
        u_inf, v_inf = self.model.forward_w(w_inf_x, w_inf_y)
        
        # Value Penalty: F(infinity) = 0
        loss_val = (u_inf**2 + v_inf**2).mean()
        
        # Derivative Penalty: dF/dw = 0 at w=1
        # Because the function is analytic, enforcing the real partial derivative 
        # is sufficient to enforce the full complex derivative.
        du_dwx = torch.autograd.grad(
            outputs=u_inf, 
            inputs=w_inf_x, 
            grad_outputs=torch.ones_like(u_inf), 
            create_graph=True
        )[0]
        
        dv_dwx = torch.autograd.grad(
            outputs=v_inf, 
            inputs=w_inf_x, 
            grad_outputs=torch.ones_like(v_inf), 
            create_graph=True
        )[0]
        
        loss_deriv = (du_dwx**2 + dv_dwx**2).mean()
        
        # Total asymptotic loss
        return loss_val + loss_deriv


    def _loss_analyticity(self, q2_re, q2_im):
        """
        Enforces Cauchy-Riemann equations: dU/dx = dV/dy and dU/dy = -dV/dx
        """
        q2_re = q2_re.clone().detach().requires_grad_(True)
        q2_im = q2_im.clone().detach().requires_grad_(True)
        
        u, v = self.model.forward(q2_re, q2_im)
        
        # Gradients of U
        grad_u_x = torch.autograd.grad(u.sum(), q2_re, create_graph=True)[0]
        grad_u_y = torch.autograd.grad(u.sum(), q2_im, create_graph=True)[0]
        
        # Gradients of V
        grad_v_x = torch.autograd.grad(v.sum(), q2_re, create_graph=True)[0]
        grad_v_y = torch.autograd.grad(v.sum(), q2_im, create_graph=True)[0]
        
        # Cauchy-Riemann Loss
        loss_cr = ((grad_u_x - grad_v_y)**2).mean() + ((grad_u_y + grad_v_x)**2).mean()
        return loss_cr


    def _loss_analyticity_w(self, device):
        """
        Enforces Cauchy-Riemann equations directly in the conformal w-disk.
        This completely avoids the physical branch cut in q^2, as the cut
        is safely mapped to the boundary |w| = 1.
        """
        n_collocation = 1000
        
        # 50% Uniform Area (global coverage)
        # 50% Uniform Radius (naturally clusters heavily at the origin)
        r_area = torch.sqrt(torch.rand(n_collocation // 2, device=device)) * 0.999
        r_center = torch.rand(n_collocation - (n_collocation // 2), device=device) * 0.999
        r = torch.cat([r_area, r_center])
        
        theta = torch.rand(n_collocation, device=device) * 2.0 * math.pi
        
        w_x = (r * torch.cos(theta)).requires_grad_(True)
        w_y = (r * torch.sin(theta)).requires_grad_(True)
        
        # Evaluate model directly in w-space
        u, v = self.model.forward_w(w_x, w_y)
        
        # Compute gradients w.r.t w_x and w_y
        grad_u_x = torch.autograd.grad(u.sum(), w_x, create_graph=True)[0]
        grad_u_y = torch.autograd.grad(u.sum(), w_y, create_graph=True)[0]
        
        grad_v_x = torch.autograd.grad(v.sum(), w_x, create_graph=True)[0]
        grad_v_y = torch.autograd.grad(v.sum(), w_y, create_graph=True)[0]
        
        # Cauchy-Riemann Loss
        loss_cr = ((grad_u_x - grad_v_y)**2).mean() + ((grad_u_y + grad_v_x)**2).mean()
        return loss_cr


    def _loss_dispersion_w(self, device):
        """
        Global Dispersion Relation evaluated entirely in the Conformal Disk.
        Uses the Poisson Integral Formula to reconstruct the interior real part
        from the boundary imaginary part. Perfectly stable, requires no subtractions.
        """
        # Sample the boundary (the physical branch cut)
        theta = torch.linspace(1e-4, math.pi - 1e-4, 1000, device=device)
        d_theta = theta[1] - theta[0]
        
        # Coordinates on the unit circle
        w_cut_x = torch.cos(theta)
        w_cut_y = torch.sin(theta)
        
        # Get Im[F] exactly on the boundary AND FLATTEN IT
        _, v_boundary = self.model.forward_w(w_cut_x, w_cut_y)
        v_boundary = v_boundary.squeeze(-1)
        
        # Sample the interior real axis (space-like region)
        # Instead of uniform sampling, use a cubic power-law to heavily 
        # cluster points near the center (low Q^2), where the physics is most precise.
        t = torch.linspace(0.0, 1.0, 100, device=device)
        
        # t**3 stays very small for a long time, then rapidly shoots up to 1.0
        r_eval = 0.05 + 0.80 * (t**3) 
        
        u_interior, _ = self.model.forward_w(r_eval, torch.zeros_like(r_eval))
        u_interior = u_interior.squeeze(-1)
        
        # Get Re[F] inside the disk directly from the network AND FLATTEN IT
        u_interior, _ = self.model.forward_w(r_eval, torch.zeros_like(r_eval))
        u_interior = u_interior.squeeze(-1)
        
        # Reconstruct Re[F] using the Poisson Integral Formula
        # Expand dimensions for PyTorch broadcasting
        r_mat = r_eval.unsqueeze(1)
        theta_mat = theta.unsqueeze(0)
        v_mat = v_boundary.unsqueeze(0)
        
        numerator = 2.0 * r_mat * torch.sin(theta_mat)
        denominator = 1.0 + r_mat**2 - 2.0 * r_mat * torch.cos(theta_mat)
        kernel = numerator / denominator
        
        # Numerical integration (Riemann sum)
        integral = torch.sum(v_mat * kernel * d_theta, dim=1)
        
        # U(r) = 1.0 - (1/pi) * Integral
        u_reconstructed = 1.0 - (1.0 / math.pi) * integral  # Output shape: [50]
        
        # Mean Squared Error between network's internal prediction and boundary reconstruction
        return ((u_interior - u_reconstructed)**2).mean()


    # def _loss_dispersion_twice_w(self, q2_eval, device):
    #     """
    #     Evaluates the exact twice-subtracted dispersion relation, but performs
    #     the numerical integration over the compact conformal boundary [-pi, 0]
    #     instead of integrating to infinity. 
    #     This completely eliminates the tug-of-war with spacelike data.
    #     """
    #     # Ensure flat 1D tensor
    #     s = q2_eval.clone().detach().view(-1)
        
    #     # Physics constants
    #     c_sq = FOUR_M_PI2
    #     r2_pi_gev2 = self.r2_pi * CONV_FM2_GEV2
        
    #     # Sample the physical branch cut in conformal angle [-pi, 0]
    #     # Skip exactly -pi (threshold) and 0 (infinity) to avoid divide-by-zero
    #     theta = torch.linspace(-math.pi + 1e-4, -1e-4, 1000, device=device)
    #     d_theta = theta[1] - theta[0]
        
    #     w_cut_x = torch.cos(theta)
    #     w_cut_y = torch.sin(theta)
        
    #     # Get Im[F] exactly on the boundary
    #     _, v_boundary = self.model.forward_w(w_cut_x, w_cut_y)
    #     v_boundary = v_boundary.squeeze()
        
    #     # Reconstruct the integral using the Exact Variable Substitution
    #     # Expand dims for broadcasting: s is [N, 1], theta is [1, 1000]
    #     s_mat = s.unsqueeze(1)
    #     theta_mat = theta.unsqueeze(0)
    #     v_mat = v_boundary.unsqueeze(0)
        
    #     cos_t = torch.cos(theta_mat)
    #     sin_t = torch.sin(theta_mat)
        
    #     # Transformed Kernel = (sin(theta) * (1 - cos(theta))) / (2c^2 - s * (1 - cos(theta)))
    #     numerator = sin_t * (1.0 - cos_t)
    #     denominator = 2.0 * c_sq - s_mat * (1.0 - cos_t)
        
    #     kernel = numerator / denominator
        
    #     # Integral = sum( V * Kernel * d_theta )
    #     integral = torch.sum(v_mat * kernel * d_theta, dim=1)
        
    #     # Multiply by the transformed prefactor: -s^2 / (2 * pi * c^2)
    #     prefactor = -(s**2) / (2.0 * math.pi * c_sq)
    #     disp_integral = prefactor * integral
        
    #     # Add the twice-subtracted anchors
    #     # Re[F(s)] = 1.0 + (1/6)<r^2>s + Integral
    #     u_dispersive = 1.0 + (1.0 / 6.0) * r2_pi_gev2 * s + disp_integral
        
    #     # Get the network's actual prediction for the real part at s
    #     u_network, _ = self.model.forward(s, torch.zeros_like(s))
    #     u_network = u_network.squeeze()
        
    #     return ((u_network - u_dispersive)**2).mean()

    def _loss_dispersion_twice_w(self, q2_eval, device):
        """
        Evaluates the exact twice-subtracted dispersion relation using a 
        SELF-CONSISTENT subtraction constant. It extracts the network's own 
        derivative at q^2=0 instead of relying on an experimental hardcoded value.
        """
        # Ensure flat 1D tensor
        s = q2_eval.clone().detach().view(-1)
        c_sq = FOUR_M_PI2
        
        q2_0 = torch.tensor([0.0], device=device, requires_grad=True)
        u_0, _ = self.model.forward(q2_0, torch.zeros_like(q2_0))
        F_prime_0 = torch.autograd.grad(u_0.sum(), q2_0, create_graph=True)[0].squeeze()
        
        # Sample the physical branch cut in conformal angle [-pi, 0]
        theta = torch.linspace(-math.pi + 1e-4, -1e-4, 1000, device=device)
        d_theta = theta[1] - theta[0]
        
        w_cut_x = torch.cos(theta)
        w_cut_y = torch.sin(theta)
        
        _, v_boundary = self.model.forward_w(w_cut_x, w_cut_y)
        v_boundary = v_boundary.squeeze()
        
        # Reconstruct the integral using the Exact Variable Substitution
        s_mat = s.unsqueeze(1)
        theta_mat = theta.unsqueeze(0)
        v_mat = v_boundary.unsqueeze(0)
        
        cos_t = torch.cos(theta_mat)
        sin_t = torch.sin(theta_mat)
        
        numerator = sin_t * (1.0 - cos_t)
        denominator = 2.0 * c_sq - s_mat * (1.0 - cos_t)
        
        kernel = numerator / denominator
        integral = torch.sum(v_mat * kernel * d_theta, dim=1)
        
        prefactor = -(s**2) / (2.0 * math.pi * c_sq)
        disp_integral = prefactor * integral
        
        # Re[F(s)] = 1.0 + F'(0)*s + Integral
        u_dispersive = 1.0 + (F_prime_0 * s) + disp_integral
        
        u_network, _ = self.model.forward(s, torch.zeros_like(s))
        u_network = u_network.squeeze()
        
        return ((u_network - u_dispersive)**2).mean()

    def _loss_nth_moment_sum_rule_w(self, n, device):
        """
        Forces the network's local n-th derivative F^(n)(0) to exactly match
        the global dispersive sum rule for that specific moment.
        Valid for any moment n >= 1 (where n=1 is the charge radius slope, 
        n=2 is curvature, n=3 is 6th moment, etc.)
        """
        # Get the Local n-th Derivative via iterative Autograd
        q2_0 = torch.tensor([0.0], device=device, requires_grad=True)
        q2_im_0 = torch.tensor([0.0], device=device)
        
        u_0, _ = self.model.forward(q2_0, q2_im_0)
        
        current_derivative = u_0.sum()
        for _ in range(n):
            # allow_unused=True safely catches instances where the network naturally 
            # flattens out at higher derivatives and disconnects the computational graph.
            grad_tuple = torch.autograd.grad(
                outputs=current_derivative, 
                inputs=q2_0, 
                create_graph=True, 
                allow_unused=True
            )
            
            if grad_tuple[0] is None:
                current_derivative = torch.zeros_like(q2_0)
                break
                
            current_derivative = grad_tuple[0].sum()
            
        F_n_local = current_derivative
        
        # Get the Global n-th Derivative via Conformal Boundary Integral
        theta = torch.linspace(-math.pi + 1e-4, -1e-4, 1000, device=device)
        d_theta = theta[1] - theta[0]
        
        w_cut_x = torch.cos(theta)
        w_cut_y = torch.sin(theta)
        
        _, v_boundary = self.model.forward_w(w_cut_x, w_cut_y)
        v_boundary = v_boundary.squeeze()
        
        c_sq = FOUR_M_PI2
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Transformed integrand for 1/s^(n+1) in conformal space
        numerator = ((1.0 - cos_t)**(n - 1)) * (-sin_t)
        denominator = (2.0**n) * (c_sq**n)
        
        integrand = v_boundary * (numerator / denominator)
        integral = torch.sum(integrand * d_theta)
        
        # F^(n)(0) = (n! / pi) * Integral
        F_n_global = (math.factorial(n) / math.pi) * integral
        
        # Scale the error by the magnitude of the global moment.
        # We use .detach() so the denominator acts strictly as a numerical 
        # scaling constant and does not distort the gradient flow of the integral.
        scale = torch.abs(F_n_global.detach()) + 1e-8
        return (((F_n_local.squeeze() - F_n_global) / scale)**2).mean()


    def compute_all_losses(self, alpha_phys, spacelike_loader, timelike_loader, phase_interpolator, device, print_losses=True):
        total_loss = torch.tensor(0.0, device=device)
        
        L_analyticity = self._loss_analyticity_w(device=device)
        total_loss += self.loss_weights['analyticity'] * alpha_phys * L_analyticity

        L_asymptotics = self._loss_asymptotics(device)
        total_loss += self.loss_weights['asymptotics'] * L_asymptotics
        
        L_alpha_asymptotic = self._loss_pqcd(device)
        total_loss += self.loss_weights['pqcd'] * L_alpha_asymptotic

        q2_watson = torch.linspace(FOUR_M_PI2, 1.0, 1000, device=device)
        L_watson = self._loss_watson(q2_watson, phase_interpolator, device)
        total_loss += self.loss_weights['watson'] * L_watson

        L_positivity = self._loss_positivity_w(device)
        total_loss += self.loss_weights['positivity'] * L_positivity
        
        q2_disp = torch.linspace(-3.0, 0.0, 1000, device=device)
        L_dispersion = self._loss_dispersion_twice_w(q2_disp, device=device)
        total_loss += self.loss_weights['dispersion'] * alpha_phys * L_dispersion

        moment_range = range(1, 4) 
        L_moments_total = torch.tensor(0.0, device=device)
        for n in moment_range:
            L_moments_total += 1/n * self._loss_nth_moment_sum_rule_w(n=n, device=device)
        total_loss += self.loss_weights['moments'] * alpha_phys * L_moments_total

        L_monotonicity = self._loss_monotonicity_w(device=device)
        total_loss += self.loss_weights['monotonicity'] * L_monotonicity

        L_spacelike = torch.tensor(0.0, device=device)
        for q2_exp, F_exp, weight_exp in spacelike_loader:
            u_pred, v_pred = self.model.forward(q2_exp)

            # Fractional MSE (Relative Error)
            # No Q^2 multiplication needed. The denominator naturally flattens the scale.
            L_batch = ((((u_pred - F_exp) / F_exp)**2) * weight_exp).mean()

            L_v_penalty = (v_pred**2).mean() 
            L_spacelike += (L_batch + L_v_penalty)
        total_loss += self.loss_weights['spacelike_data'] * L_spacelike

        L_timelike = torch.tensor(0.0, device=device)
        for q2_exp, alpha_exp, Fpi_sq_exp, weight_exp in timelike_loader:
            u_pred, v_pred = self.model.forward_mixed(q2_exp, 1-alpha_exp)

            Fpi_sq_pred = u_pred**2 + v_pred**2
            L_batch = ((Fpi_sq_pred - Fpi_sq_exp)**2 * weight_exp).mean()
            L_timelike += L_batch
        total_loss += self.loss_weights['timelike_data'] * L_timelike

        # Apply a Gaussian prior to the mixing parameters
        # Assuming alpha ~ 0.002 +- 0.001 and phi_omega ~ 0 +- 0.25 rad (mixing parameter is nearly real)
        L_prior_alpha = ((self.model.alpha - 0.002) / 0.001)**2
        L_prior_phi = (self.model.phi_omega / 0.25)**2
        total_loss += self.loss_weights['mixing'] * (L_prior_alpha + L_prior_phi).squeeze()

        # Verbose Output
        if print_losses:
            print(f"Total Loss: {total_loss.item():.4f}")
            print(f"  Analyticity: {L_analyticity.item():.4f}, Moments: {L_moments_total.item():.4f}, Watson: {L_watson.item():.4f}, Monotonicity: {L_monotonicity.item():.4f}")
            print(f"  Positivity: {L_positivity.item():.4f}, Asymp: {L_asymptotics.item():.4f}, PQCD: {L_alpha_asymptotic.item():.4f}, Dispersion: {L_dispersion.item():.4f}")
            print(f"  L_spacelike: {L_spacelike.item():.4f}, L_timelike: {L_timelike.item():.4f}")

        return total_loss


    def train(self, n_epochs, lr, continue_training, spacelike_loader, timelike_loader, phase_interpolator, device, verbose=True):
        self.optimizer = torch.optim.Adam([
            {'params': self.model.input_layer.parameters()},
            {'params': self.model.hidden_blocks.parameters()},
            {'params': self.model.output_layer.parameters()},
            # Force the mixing parameters to learn much faster than the MLP
            {'params': [self.model.alpha, self.model.phi_omega], 'lr': lr * 50.0} 
        ], lr=lr)
        best_loss = float('inf')

        for epoch in range(n_epochs):
            if (continue_training == 0):
                alpha_phys = 0.0
            elif (continue_training == 1):
                alpha_phys = 1.0
            else:
                alpha_phys = epoch / n_epochs  # Linear ramp from 0 to 1

            print_losses = False
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch}/{n_epochs}")
                print_losses = True

            total_loss = self.compute_all_losses(alpha_phys, spacelike_loader, timelike_loader, phase_interpolator, device, print_losses) 
            if total_loss < best_loss:
                best_loss = total_loss.item()
                self.best_param = copy.deepcopy(self.model.state_dict())
                self.best_opt_param = copy.deepcopy(self.optimizer.state_dict())

            # Optimization Step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # Save the absolute best loss value to the class instance
        self.best_loss_val = best_loss
        print(f"Training finished after {n_epochs} epochs.")
        return best_loss


    def fine_tune_lbfgs(self, n_steps, lr, spacelike_loader, timelike_loader, phase_interpolator, device, verbose=True):
        """
        Executes L-BFGS optimization for final precise convergence.
        Requires the FULL dataset passed as single tensors/lists, not mini-batches.
        """
        print("--- Initiating L-BFGS Fine-Tuning ---")
        
        # Initialize L-BFGS with standard PINN parameters
        lbfgs_optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=lr,                 # Default learning rate is 1.0 (acts as a step length)
            max_iter=20,           # Max iterations per optimization step
            max_eval=25,           # Max function evaluations per step
            history_size=100,       # Size of the memory for the Hessian approximation
            tolerance_grad=1e-7,   # Termination tolerance on first order optimality
            tolerance_change=1e-9, # Termination tolerance on function value/parameter changes
            line_search_fn="strong_wolfe" # Highly recommended to ensure sufficient loss decrease
        )
        
        step_counter = 0
        def closure():
            """
            The closure function clears gradients, computes the full forward pass,
            calculates all loss components, and computes the backward pass.
            """
            nonlocal step_counter
            lbfgs_optimizer.zero_grad()
            
            total_loss = self.compute_all_losses(1, spacelike_loader, timelike_loader, phase_interpolator, device, print_losses=False)
            total_loss.backward()
            
            if total_loss.item() < self.best_loss_val:
                self.best_loss_val = total_loss.item()
                self.best_param = copy.deepcopy(self.model.state_dict())
                self.best_opt_param = copy.deepcopy(lbfgs_optimizer.state_dict())

            if verbose and step_counter % 10 == 0:
                print(f"L-BFGS Step {step_counter} - Total Loss: {total_loss.item():.6f}")
                
            step_counter += 1
            return total_loss

        # Optimization Loop
        for _ in range(n_steps):
            lbfgs_optimizer.step(closure)
            
        print("L-BFGS fine-tuning complete.")


    def save_checkpoint(self, path="pinn_checkpoint.pt", lr=1e-3):
        """Saves the current state of the model and optimizer."""

        torch.save({
            'model_state_dict': self.best_param,
            'optimizer_state_dict': self.best_opt_param,
        }, path)
        print(f"Checkpoint saved successfully to {path}")


    def load_checkpoint(self, path="pinn_checkpoint.pt"):
        """Loads the saved state of the model and optimizer to resume training."""
        if not os.path.exists(path):
            print(f"Checkpoint file not found at {path}. Starting training from scratch.")
            return False
            
        # Ensure optimizer is initialized before loading state
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        # The checkpoint may have been saved on another device (e.g. CUDA)
        checkpoint = torch.load(path, map_location=next(self.model.parameters()).device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Checkpoint loaded successfully from {path}. Model ready to resume training.")
        return True
