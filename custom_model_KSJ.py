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

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

# ============================================================================
# GENERATIVE MODEL SKELETON
# ============================================================================

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

        beta_start = kwargs.get('beta_start', 1e-4)
        beta_end = kwargs.get('beta_end', 0.02)
        beta_mode = kwargs.get('beta_mode', 'quad')

        if beta_mode == "linear":
            betas = torch.linspace(beta_start, beta_end, steps=num_train_timesteps)
        elif beta_mode == "quad":
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
            )

        alphas = 1-betas
        alphas_cumprod = alphas.cumprod(dim=0)
        # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
    
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

        alphas_prod_t = extract(self.alphas_cumprod, t, data)
        xt = data*torch.sqrt(alphas_prod_t) + torch.sqrt(1-alphas_prod_t)*noise
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
    #     if isinstance(t, int):
    #         t = torch.tensor([t]).to(self.device)
    #     eps_factor = (1 - extract(self.var_scheduler.alphas, t, xt)) / (
    #         1 - extract(self.var_scheduler.alphas_cumprod, t, xt)
    #     ).sqrt()
    #     eps_theta = self.network(xt, t)

    #     mu=(xt-eps_factor*eps_theta)/extract(self.var_scheduler.alphas, t, xt).sqrt()

    #     alpha_prod_t_prev = extract(self.var_scheduler.alphas_cumprod_prev, t, xt)
    #     if (t==0).all():
    #       x_t_prev = mu
    #     else:
    #       var = (extract(self.var_scheduler.betas, t, xt)*(1-alpha_prod_t_prev)/(1-extract(self.var_scheduler.alphas_cumprod, t, xt)))
    #       clampped_var = torch.clamp(var, min=1e-20)
    #       x_t_prev = mu + clampped_var.sqrt() * torch.randn_like(xt)
    #     return x_t_prev
        alpha_prod_t = extract(self.alphas_cumprod, t, xt)
        x0 = (xt - (1 - alpha_prod_t).sqrt() * pred) / alpha_prod_t.sqrt()
        x0 = torch.clamp(x0, -1.0, 1.0)
        mask = (t_next < 0)
        alpha_prod_t_prev_safe = extract(self.alphas_cumprod, t_next.clamp(min=0), xt)
        alpha_prod_t_prev = torch.where(
            mask.view(-1, 1, 1, 1), 
            torch.ones_like(alpha_prod_t), 
            alpha_prod_t_prev_safe
        )
        addi_var = eta**2 * extract(self.betas, t, xt) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        addi_var = torch.nan_to_num(addi_var, nan=0.0, posinf=0.0, neginf=0.0)
        dir_term_coeff_sq = 1 - alpha_prod_t_prev - addi_var
        dir_term_coeff = torch.sqrt(torch.clamp(dir_term_coeff_sq, min=0.0))
        mu = alpha_prod_t_prev.sqrt() * x0 + dir_term_coeff * (xt - alpha_prod_t.sqrt() * x0) / (1 - alpha_prod_t).sqrt()
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        z = torch.where(
            mask.view(-1, 1, 1, 1),
            torch.zeros_like(xt),
            torch.randn_like(xt)  
        )
        x_t_prev = mu + addi_var.sqrt() * z
        return x_t_prev
    
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
        return noise


class CustomGenerativeModel(BaseGenerativeModel):
    """
    Custom Generative Model Skeleton
    
    Students need to implement this class by inheriting from BaseGenerativeModel.
    This class wraps the network and scheduler to provide training and sampling interfaces.
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
        # TODO: Initialize your model-specific parameters (e.g., EMA, loss weights)
    
    def compute_loss(self, data, noise, **kwargs):
        """
        Compute the training loss.
        
        Args:
            data: Clean data batch
            noise: Noise batch (or x0 for flow models)
            **kwargs: Additional arguments
            
        Returns:
            Loss tensor
        """
        t = self.scheduler.sample_timesteps(data.shape[0], device=data.device)
        xt = self.scheduler.forward_process(data, noise, t)
        pred_noise = self.predict(xt, t, **kwargs)
        target = self.scheduler.get_target(data, noise, t)
        loss=(((pred_noise-target).pow(2))).mean()
        return loss
    
    def predict(self, xt, t, **kwargs):
        """
        Make prediction given noisy data and timestep.
        
        Args:
            xt: Noisy data
            t: Timestep
            **kwargs: Additional arguments (e.g., condition for additional timestep)
            
        Returns:
            Model prediction
        """
        return self.network(xt, t, **kwargs)
    
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

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        device = self.device
        eta = kwargs.get('eta', 0.0)
        xt = torch.randn(shape, device=device)
        traj = [xt] if return_traj else None
        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 
            0, 
            num_inference_timesteps, 
            device=device
        ).long()
        loop = tqdm(range(num_inference_timesteps), desc="Sampling", disable=not verbose, leave=False)
        for i in loop:
            t_val = timesteps[i]
            t_next_val = timesteps[i+1] if i < num_inference_timesteps - 1 else -1
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            t_next = torch.full((shape[0],), t_next_val, device=device, dtype=torch.long)
            pred_noise = self.predict(xt, t, **kwargs)
            xt = self.scheduler.reverse_process_step(xt, pred_noise, t, t_next, eta=eta)
            if return_traj:
                traj.append(xt)

        return traj if return_traj else xt


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
    
    # Create your model
    model = CustomGenerativeModel(network, scheduler, **kwargs)
    
    return model.to(device)