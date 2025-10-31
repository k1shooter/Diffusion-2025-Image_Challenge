#!/usr/bin/env python3
"""
Template for implementing custom generative models
Students should create their own implementation by inheriting from the base classes.

This file provides skeleton code for implementing generative models.
Students need to implement the TODO sections in their own files.
"""

import torch
from src.base_model import BaseScheduler, BaseGenerativeModel
from src.network import UNet
from tqdm import tqdm
import copy
import torch.nn.functional as F
import numpy as np
import math

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# ============================================================================
# GENERATIVE MODEL SKELETON
# ============================================================================
# Helper function from OpenAI code (cm/nn.py)
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

class CustomScheduler(BaseScheduler):
    """
    Custom Scheduler Skeleton
    
    TODO: Students need to implement this class in their own file.
    Required methods:
    1. sample_timesteps: Sample random timesteps for training
    2. forward_process: Apply forward process to transform data
    3. reverse_process_step: Perform one step of the reverse process
    4. get_target: Get target for model prediction
    """
    
    def __init__(self, num_train_timesteps: int = 1000, **kwargs):
        super().__init__(num_train_timesteps, **kwargs)
        # TODO: Initialize your scheduler-specific parameters (e.g., betas, alphas, sigma_min)
        self.num_train_timesteps = num_train_timesteps

        self.sigma_min = kwargs.get('sigma_min', 0.002)  # ε [cite: 72]
        self.sigma_max = kwargs.get('sigma_max', 80.0)    # T [cite: 72]
        self.rho = kwargs.get('rho', 7.0)                 # ρ [cite: 140]
        self.sigma_data = kwargs.get('sigma_data', 0.5)   # σ_data [cite: 1229]
        
        # Create the time discretization
        # t_i = (ε^(1/ρ) + (i/(N-1)) * (T^(1/ρ) - ε^(1/ρ)))^ρ  for i=0...N-1 [cite: 140]
        timesteps_rho = torch.linspace(0, 1, num_train_timesteps) # Range [0, 1]
        sigmas = (
            self.sigma_max**(1 / self.rho) + \
            timesteps_rho * \
            (self.sigma_min**(1 / self.rho) - self.sigma_max**(1 / self.rho))
        )**self.rho
        
        # Store sigmas (noise levels)
        self.register_buffer('sigmas', sigmas.float())
    
    def sample_timesteps(self, batch_size: int, device: torch.device):
        """
        Sample random timesteps for training.
        
        Returns:
            Tensor of shape (batch_size,) with timestep values
        """
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
    
    def forward_process(self, data, noise, t):
        """
        Apply forward process to add noise to clean data.
        
        Args:
            data: Clean data tensor
            noise: Noise tensor
            t: Timestep tensor
            
        Returns:
            Noisy data at timestep t
        """

        sigmas_t = t.view(-1, 1, 1, 1) if t.ndim == 1 else t
        xt = data + sigmas_t * noise
        return xt

    
    def reverse_process_step(self, xt, pred, t, t_next, eta=0.0):
        """
        Perform one step of the reverse (denoising) process.
        
        Args:
            xt: Current noisy data
            pred: Model prediction (e.g., predicted noise, velocity, or x0)
            t: Current timestep
            t_next: Next timestep
            
        Returns:
            Updated data at timestep t_next
        """
        if t_next == self.sigma_min:
            return pred

        # Calculate noise standard deviation for the next step
        variance = (t_next**2 - self.sigma_min**2).clamp(min=1e-8) # Add small epsilon for stability
        noise_std = torch.sqrt(variance)

        z = torch.randn_like(pred)
        # Add noise
        xt_next = pred + append_dims(noise_std, pred.ndim) * z
        return xt_next

    
    def get_target(self, data, noise, t):
        """
        Get the target for model prediction (what the network should learn to predict).
        
        Args:
            data: Clean data
            noise: Noise
            t: Timestep
            
        Returns:
            Target tensor (e.g., noise for DDPM, velocity for Flow Matching)
        """
        return data


class CustomGenerativeModel(BaseGenerativeModel):
    """
    Custom Generative Model Skeleton
    
    Students need to implement this class by inheriting from BaseGenerativeModel.
    This class wraps the network and scheduler to provide training and sampling interfaces.
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
        # TODO: Initialize your model-specific parameters (e.g., EMA, loss weights)
        # Create the target network (f_θ⁻) 
        # This is a deep copy of the online network (f_θ)
        self.target_network = copy.deepcopy(network)
        self.target_network.requires_grad_(False)

        # 2. [추가] 추론용 EMA 네트워크 (샘플링/저장용)
        self.ema_network = copy.deepcopy(network)
        self.ema_network.requires_grad_(False)
        
        # 3. [추가] EMA Decay Rate (Table 3 )
        self.ema_decay_rate = kwargs.get('ema_decay_rate', 0.9999)

        self.s0 = kwargs.get('s0', 2.0)       # Initial discretization steps [cite: 1241]
        self.s1 = kwargs.get('s1', 150.0)    # Target discretization steps [cite: 1241]
        self.mu0 = kwargs.get('mu0', 0.9)
        
        # # EMA decay rate (μ in Alg 3) 
        # self.ema_decay = kwargs.get('ema_decay', 0.999) # Default, can be tuned
        
        self.loss_type = kwargs.get('loss_type', 'lpips' if LPIPS_AVAILABLE else 'l2')
        if self.loss_type == 'l2':
            self.loss_fn = F.mse_loss
            print("Consistency Model initialized with L2 loss. WARNING: This might lead to blurry results or noise. LPIPS is strongly recommended.")
        elif self.loss_type == 'l1':
            self.loss_fn = F.l1_loss
            print("Consistency Model initialized with L1 loss.")
        elif self.loss_type == 'lpips':
            if not LPIPS_AVAILABLE:
                print("Warning: lpips package not found despite being requested. Defaulting to L2 loss.")
                self.loss_type = 'l2'
                self.loss_fn = F.mse_loss
            else:
                # Initialize LPIPS model on the same device as the network
                lpips_net = lpips.LPIPS(net='vgg').to(next(network.parameters()).device)
                # Freeze LPIPS model parameters
                for param in lpips_net.parameters():
                    param.requires_grad = False
                self.lpips_fn = lpips_net
                # LPIPS loss function
                self.loss_fn = lambda a, b: self.lpips_fn(a, b).mean()
                print(f"Consistency Model initialized with LPIPS loss.")

        # --- Store sigma_data for scaling ---
        self.sigma_data = self.scheduler.sigma_data

    @torch.no_grad()
    def update_ema(self, k: int, K: int):
        """
        Performs the EMA update for the target network (f_θ⁻).
        This MUST be called in the training loop after optimizer.step().
        """
        # Calculate N(k) using the same formula as in compute_loss 
        current_N = math.floor(math.sqrt((k / K) * (self.s1**2 - self.s0**2) + self.s0**2)) + 1
        
        # Calculate adaptive μ(k) 
        # μ(k) = exp(s0 * log(μ0) / N(k))
        current_mu = math.exp(self.s0 * math.log(self.mu0) / current_N)
        
        online_params = dict(self.network.named_parameters())
        target_params = dict(self.target_network.named_parameters())

        for name, param in online_params.items():
            if name in target_params:
                target_params[name].data.mul_(current_mu).add_(param.data, alpha=1 - current_mu)

    @torch.no_grad()
    def update_model_ema(self):
        """
        Performs Polyak averaging on the online network (f_θ) weights 
        to update the inference network (f_θ_ema).
        
        """
        decay = self.ema_decay_rate
        
        online_params = dict(self.network.named_parameters())
        ema_params = dict(self.ema_network.named_parameters())

        for name, param in online_params.items():
            if name in ema_params:
                ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)

    def _get_scalings(self, sigma):
        """
        Calculates c_skip(t) and c_out(t) from Appendix C [cite: 1230-1234].
        These satisfy the boundary condition f(x, ε) = x[cite: 1232].
        """
        
        sigma_data = self.sigma_data
        # Ensure sigma is a tensor and on the correct device
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=next(self.network.parameters()).device)
        elif sigma.device != next(self.network.parameters()).device:
             sigma = sigma.to(next(self.network.parameters()).device)
             
        sigma_view = append_dims(sigma, 4) # Add channel, height, width dims
        epsilon = self.scheduler.sigma_min # Use sigma_min as epsilon

        # c_skip(σ) = σ_data² / ((σ - ε)² + σ_data²)
        c_skip_denominator = (sigma_view - epsilon)**2 + sigma_data**2
        c_skip = (sigma_data**2) / c_skip_denominator

        # c_out(σ) = σ_data * (σ - ε) / sqrt(σ_data² + σ²)
        # Note the different denominator compared to c_skip!
        c_out_denominator = torch.sqrt(sigma_data**2 + sigma_view**2)
        c_out = sigma_data * (sigma_view - epsilon) / c_out_denominator

        return c_skip, c_out

    

    @torch.no_grad()
    def predict_target(self, xt, sigma, **kwargs):
        """
        Makes a prediction using the target network f_θ⁻.
        Used in the CT loss function.
        """
        if not isinstance(sigma, torch.Tensor):
             sigma = torch.full((xt.shape[0],), sigma, device=xt.device, dtype=xt.dtype)
        elif sigma.ndim == 0:
             sigma = sigma.repeat(xt.shape[0])
        elif sigma.shape[0] != xt.shape[0]:
             raise ValueError("Sigma tensor batch size must match xt batch size")
             
        c_skip, c_out = self._get_scalings(sigma)
        
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in_view = append_dims(c_in, xt.ndim)
        rescaled_t = 1000 * 0.25 * torch.log(sigma + 1e-44)

        F_theta_minus = self.target_network(c_in_view * xt, rescaled_t, **kwargs)
        x0_pred = c_skip * xt + c_out * F_theta_minus
        return x0_pred
    
    def compute_loss(self, data, noise, k: int, K: int, **kwargs):
        """
        Compute the training loss.
        
        Args:
            data: Clean data batch
            noise: Noise batch (or x0 for flow models)
            **kwargs: Additional arguments
            
        Returns:
            Loss tensor
        """
        """
        Computes the Consistency Training (CT) loss (Eq. 10).
        Matches `consistency_losses` logic in OpenAI's `karras_diffusion.py`
        when `training_mode == "consistency_training"`.
        """
        x_clean = data
        z = noise # Use the provided noise as z ~ N(0, I) [cite: 198]
        batch_size = x_clean.shape[0]
        device = x_clean.device

        # --- 1. Implement N(k) schedule from Appendix C --- 
        # N(k) = floor(sqrt( (k/K) * (s1^2 - s0^2) + s0^2 )) + 1
        s0_sq = self.s0**2
        s1_sq = self.s1**2
        k_over_K = k / K
        
        current_N_float = math.sqrt(k_over_K * (s1_sq - s0_sq) + s0_sq)
        current_N = math.floor(current_N_float) + 1 # 
        current_N = min(current_N, self.scheduler.num_train_timesteps) # Cap at max steps
        current_N = max(current_N, 2) # Must have at least 2 steps (n and n+1)

        # --- 2. Get the current coarse-grained sigma schedule ---
        # We sub-sample the main 1000-step schedule to get N(k) steps
        
        # Create N(k) indices from 0 to 999
        indices = torch.linspace(
            0, # Start index (maps to sigma_max)
            self.scheduler.num_train_timesteps - 1, # End index (maps to sigma_min)
            current_N, # Number of steps = N(k)
            device=device
        ).long()
        
        # Get the sigmas for these N(k) steps
        all_sigmas = self.scheduler.sigmas
        current_schedule_sigmas = all_sigmas[indices] # Shape [N(k)], sorted [T, ..., ε]

        # --- 3. Sample n ~ U[0, N(k)-2] ---
        # This is U[1, N(k)-1] in Alg 3[cite: 197], but we use 0-based indexing
        # We sample an *index* n, such that we can access n and n+1
        n = torch.randint(0, current_N - 1, (batch_size,), device=device)

        # 4. Get adjacent sigmas t_{n+1} and t_n from the *coarse* schedule
        # Note: current_schedule_sigmas is sorted [T, ..., ε]
        # So sigma_n_plus_1 is the *larger* sigma (more noise)
        sigma_n_plus_1 = current_schedule_sigmas[n]     # Higher sigma (e.g., sigmas[0])
        sigma_n = current_schedule_sigmas[n + 1]        # Lower sigma (e.g., sigmas[1])

        # 5. Create two noisy samples (Eq. 10[cite: 225], using sampled sigmas)
        # x_a = x_clean + σ_{n+1} * z
        # x_b = x_clean + σ_n * z
        x_a = self.scheduler.forward_process(x_clean, z, sigma_n_plus_1)
        x_b = self.scheduler.forward_process(x_clean, z, sigma_n)

        # 6. Get model outputs (Eq. 10)
        # Online network prediction at the higher noise level
        out_a = self.predict(x_a, sigma_n_plus_1) # f_θ(x_{t_{n+1}}, t_{n+1})

        # Target network prediction at the lower noise level (with no_grad)
        out_b = self.predict_target(x_b, sigma_n) # f_θ⁻(x_{t_n}, t_n) [cite: 200]

        # 7. Compute distance (loss)
        # Weighting λ(t_n) = 1 is used in OpenAI code and paper [cite: 163]
        if self.loss_type == 'lpips':
            # LPIPS(VGG) expects inputs in [-1, 1] range
            loss = self.loss_fn(out_a, out_b.detach())
        else: # l1 or l2
            loss = self.loss_fn(out_a, out_b.detach(), reduction='mean')
        
        return loss
    
    def predict(self, xt, sigma, use_ema=False, **kwargs):
        """
        Make prediction given noisy data and timestep.
        
        Args:
            xt: Noisy data
            t: Timestep
            **kwargs: Additional arguments (e.g., condition for additional timestep)
            
        Returns:
            Model prediction
        """
        """
        Predicts x0 given noisy data xt and noise level sigma.
        This implements f_θ(xt, σ) = c_skip(σ)xt + c_out(σ)F_θ(xt, σ).
        Matches `denoise` function logic in OpenAI's `karras_diffusion.py` when distillation=True.
        """
        # Ensure sigma is a tensor and on the correct device/shape
        if not isinstance(sigma, torch.Tensor):
             sigma = torch.full((xt.shape[0],), sigma, device=xt.device, dtype=xt.dtype)
        elif sigma.ndim == 0:
             sigma = sigma.repeat(xt.shape[0])
        elif sigma.shape[0] != xt.shape[0]:
             raise ValueError("Sigma tensor batch size must match xt batch size")
             
        # Get scaling factors
        c_skip, c_out = self._get_scalings(sigma)

        # --- U-Net Input Scaling and Timestep Embedding ---
        # Match OpenAI's `denoise` function: c_in = 1 / sqrt(σ² + σ_data²)
        # Match OpenAI's timestep embedding: rescaled_t = 1000 * 0.25 * log(σ)
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in_view = append_dims(c_in, xt.ndim)
        rescaled_t = 1000 * 0.25 * torch.log(sigma + 1e-44) # Add epsilon for log stability

        active_network = self.ema_network if use_ema else self.network
        
        F_theta = active_network(c_in_view * xt, rescaled_t, **kwargs)

        # Calculate final x0 prediction: c_skip * xt + c_out * F_theta
        x0_pred = c_skip * xt + c_out * F_theta
        return x0_pred


    @torch.no_grad()
    def sample(self, shape, num_inference_timesteps=20, return_traj=False, verbose=False, **kwargs):
        """
        Generate samples from noise using the reverse process.
        
        Args:
            shape: Shape of samples to generate (batch_size, channels, height, width)
            num_inference_timesteps: Number of denoising steps (NFE)
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            **kwargs: Additional arguments
            
        Returns:
            Generated samples (or trajectory if return_traj=True)
        """

        """
        Generate samples using Multi-step Consistency Sampling (Algorithm 1).
        
        Args:
            shape: (batch_size, C, H, W)
            num_inference_timesteps: Number of denoising steps (N)
        """
        device = next(self.network.parameters()).device
        generator = kwargs.get('generator', None) # Optional: for reproducible sampling

        # 1. Create the inference timestep schedule (τ_N, ..., τ_1) = (σ_max, ..., σ_min)
        # Using linspace on indices and then indexing sigmas
        t_indices = torch.linspace(
            0, # Start index (corresponds to sigma_max)
            self.scheduler.num_train_timesteps - 1, # End index (corresponds to sigma_min)
            num_inference_timesteps,
            device=device,
            dtype=torch.long
        )
        # Note: scheduler.sigmas should be [sigma_max, ..., sigma_min]
        taus = self.scheduler.sigmas[t_indices]

        # 2. Get initial noise sample x_τN ~ N(0, τ_N² I) where τ_N = σ_max
        tau_N = taus[0]
        xt = torch.randn(shape, device=device) * tau_N

        # 3. First evaluation: Get x_0 prediction from initial noise (Alg 1, line 3)
        # Pass the initial noise xt and the highest sigma tau_N
        x0_pred = self.predict(xt, tau_N, use_ema=True, **kwargs) # Now xt holds the first x0 prediction

        if return_traj:
            traj = [xt.cpu()] # Store trajectory on CPU to save GPU memory if needed

        # 4. Multi-step sampling loop (Alg 1, lines 5-9)
        # Loop through τ_{N-1}, ..., τ_1 (which correspond to taus[1:])
        iterator = tqdm(range(num_inference_timesteps - 1), desc="Consistency Sampling", disable=not verbose)
        for i in iterator:
            tau_n_plus_1 = taus[i]     # Sigma used in the previous step's prediction
            tau_n = taus[i+1]        # Target sigma for the current step

            x_noisy = self.scheduler.reverse_process_step(
                xt=0,
                pred=x0_pred,       # Pass previous x0 prediction
                t=tau_n_plus_1,
                t_next=tau_n,       # Target noise level
            )

            # Denoise using the online network at the current noise level τ_n (Alg 1, line 8)
            # Predict x_0 from x_noisy at sigma=τ_n
            x0_pred = self.predict(x_noisy, tau_n, use_ema=True, **kwargs)

            if return_traj:
                traj.append(x0_pred.cpu())

        return torch.stack(traj) if return_traj else x0_pred


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_custom_model(device="cpu", **kwargs):
    """
    Example function to create a custom generative model.
    
    Students should modify this function to create their specific model.
    
    Args:
        device: Device to place model on
        **kwargs: Additional arguments that can be passed to network or scheduler
                  (e.g., num_train_timesteps, use_additional_condition for scalar conditions
                   like step size in Shortcut Models or end timestep in Consistency Trajectory Models, etc.)
    """
    
    # Create U-Net backbone with FIXED hyperparameters
    # DO NOT MODIFY THESE HYPERPARAMETERS
    network = UNet(
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_additional_condition=kwargs.get('use_additional_condition', False)
    )
    
    # Extract scheduler parameters with defaults
    num_train_timesteps = kwargs.pop('num_train_timesteps', 1000)
    
    # Create your scheduler
    scheduler = CustomScheduler(num_train_timesteps=num_train_timesteps, **kwargs)
    
    kwargs.setdefault('ema_decay_rate', 0.9999)
    # Create your model
    model = CustomGenerativeModel(network, scheduler, **kwargs)
    
    return model.to(device)