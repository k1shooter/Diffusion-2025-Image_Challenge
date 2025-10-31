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

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand

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
        beta_mode = kwargs.get('beta_mode', 'linear')

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

        sigmas = torch.sqrt(1 - self.alphas_cumprod)
        lambdas = torch.log(self.alphas_cumprod.sqrt()) - torch.log(sigmas)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("lambdas", lambdas)
    
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

        dpm_alpha_t = extract(self.alphas_cumprod.sqrt(), t, data)
        dpm_sigma_t = extract(self.sigmas, t, data)
        xt = data*dpm_alpha_t + dpm_sigma_t*noise
        return xt
    
    def reverse_process_step(self, xt, pred, t, t_next):
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
        alpha_t = extract(self.alphas_cumprod.sqrt(), t, xt)
        #    -1 인덱싱을 방지하기 위함.
        t_next_safe = torch.clamp(t_next, min=0)

         # 3. "정상 스텝" (t_next >= 0)일 때의 값 계산
        alpha_t_prev_normal = extract(self.alphas_cumprod.sqrt(), t_next_safe, xt)
        # 4. "마지막 스텝" (t_next = -1)일 때의 값 (alpha_bar_0.sqrt() = 1.0)
        alpha_t_prev_final = torch.tensor(1.0, device=xt.device, dtype=xt.dtype)
        # 5. torch.where로 실제 사용할 alpha_t_prev 선택
        # [FIX] is_final_step의 shape을 [B]에서 [B, 1, 1, 1]로 변경하여 브로드캐스팅합니다.
        is_final_step = (t_next < 0).view(-1, 1, 1, 1)
        alpha_t_prev = torch.where(is_final_step, alpha_t_prev_final, alpha_t_prev_normal)

        # sigma_t_prev 계산
        sigma_t_prev_normal = extract(self.sigmas, t_next_safe, xt)
        # t_next = -1 (최종 스텝)일 때 sigma_t_prev는 0이 되어야 합니다.
        sigma_t_prev_final = torch.tensor(0.0, device=xt.device, dtype=xt.dtype)
        sigma_t_prev = torch.where(is_final_step, sigma_t_prev_final, sigma_t_prev_normal)

        lambda_t = extract(self.lambdas, t, xt)

        # lambda_t_prev 계산
        lambda_t_prev_normal = extract(self.lambdas, t_next_safe, xt)
        # t_next = -1일 때, t_next_safe = 0이 되므로 lambdas[0]를 사용합니다.
        # DPM-Solver 논문에 따라 lambda(0)는 매우 큰 값(inf)이 될 수 있으므로,
        # 이 경우 h = -lambda_t 가 되도록 lambda_t_prev = 0 으로 설정할 수 있습니다.
        # 하지만, 1차 DPM-Solver의 경우 t_next_safe=0 (lambdas[0]) 값을 그냥 사용해도 괜찮습니다.
        # 여기서는 lambda(0)이 유한하다고 가정합니다.
        lambda_t_prev_final = extract(self.lambdas, t_next_safe, xt) # t_next_safe가 0이므로 lambdas[0]
        lambda_t_prev = torch.where(is_final_step.view(-1), lambda_t_prev_final.view(-1), lambda_t_prev_normal.view(-1))
        lambda_t_prev = lambda_t_prev.view(-1, 1, 1, 1) # shape 복원

        h = lambda_t_prev - lambda_t

        xt_prev = (alpha_t_prev / alpha_t) * xt - (sigma_t_prev * (torch.exp(h) - 1)) * pred
        return xt_prev
    
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
        xt = torch.randn(shape, device=device)
        traj = [xt]
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
          xt = self.scheduler.reverse_process_step(xt, pred_noise, t, t_next)
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