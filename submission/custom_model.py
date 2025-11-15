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
        self.M = kwargs.get('M', 128) # Max denoising steps [cite: 1899, 2291]
        self.log2_M = int(np.log2(self.M))
    
    def sample_timesteps(self, batch_size: int, device: torch.device):
        """
        Sample random timesteps for training.
        
        Returns:
            Tensor of shape (batch_size,) with timestep values
        """
        # d = 1/self.M
        # linspace = torch.linspace(0, 1 - d, self.M, device=device)
        return torch.rand(batch_size, device=device)
    
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

        t_full = append_dims(t, data.ndim)

        xt = (1-t_full)*noise + t_full * data
        return xt

    
    def reverse_process_step(self, xt, pred, d):
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
        if isinstance(d, torch.Tensor) and d.ndim < xt.ndim:
            d = append_dims(d, xt.ndim)

        xt_next = xt + d * pred
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
        return data-noise


class CustomGenerativeModel(BaseGenerativeModel):
    """
    Custom Generative Model Skeleton
    
    Students need to implement this class by inheriting from BaseGenerativeModel.
    This class wraps the network and scheduler to provide training and sampling interfaces.
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
        # TODO: Initialize your model-specific parameters (e.g., EMA, loss weights)

        self.ema_network = copy.deepcopy(network)
        self.ema_network.requires_grad_(False)
        self.ema_decay_rate = kwargs.get('ema_decay_rate', 0.999)

        # 3. Training parameters
        self.M = self.scheduler.M # Max denoising steps (e.g., 128)
        self.log2_M = self.scheduler.log2_M
        # Ratio of Empirical (Flow-Matching) to Bootstrap (Self-Consistency) targets
        self.empirical_ratio = kwargs.get('empirical_ratio', 0.75) # [cite: 1917, 2289, 2531]
        self.bootstrap_ratio = 1.0 - self.empirical_ratio
        self.loss_fn = F.mse_loss
        print(f"Shortcut Model initialized with L2(MSE) loss. (Bootstrap ratio: {self.bootstrap_ratio})")

    @torch.no_grad()
    def update_ema(self, k: int, K: int):
        """
        Performs the EMA update for the target network (f_θ⁻).
        This MUST be called in the training loop after optimizer.step().
        """
        decay = self.ema_decay_rate
        
        online_params = dict(self.network.named_parameters())
        ema_params = dict(self.ema_network.named_parameters())
        for name, param in online_params.items():
            if name in ema_params:
                ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)

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
        x1 = data
        x0 = noise
        device = data.device
        B = data.shape[0]
        
        # Split batch for joint training
        B_bst = int(B * self.bootstrap_ratio)
        B_flow = B - B_bst
        
        total_loss = 0.0
        
        # --- 1. Flow-Matching Loss (d=0) [75% of batch] ---
        if B_flow > 0:
            x1_flow, x0_flow = data[B_bst:], noise[B_bst:]
            
            t_flow = self.scheduler.sample_timesteps(B_flow, device)
            xt_flow = self.scheduler.forward_process(x1_flow, x0_flow, t_flow)
            v_target = self.scheduler.get_target(x1_flow, x0_flow, None)
            
            d_flow = torch.zeros_like(t_flow) # d=0
            v_pred = self.predict(xt_flow, t_flow, d_flow, use_ema=False)
            
            loss_flow = self.loss_fn(v_pred, v_target)
            total_loss += self.empirical_ratio * loss_flow

        # --- 2. Self-Consistency (Bootstrap) Loss (d>0) [25% of batch] ---
        # if B_bst > 0:
        #     x1_bst, x0_bst = data[:B_bst], noise[:B_bst]

        #     # Sample d (step size) from {1/128, 1/64, ..., 1}
        #     dt_base_idx = torch.randint(0, self.log2_M, (B_bst,), device=device) # 0~6
        #     d = 1.0 / (2*2.0 ** dt_base_idx) #1/2 ~ 1/128
            
        #     # t : [0, 1 - 2d], discrete, multiple of d
        #     num_steps_in_range = (2.0 ** dt_base_idx).long() # 1  ~ 64, 1/2d
            
        #     # 각 샘플마다 유효한 범위 [0, num_steps_in_range-1]에서 정수 인덱스 샘플링
        #     t_rand_unscaled = torch.rand(B_bst, device=device) # U[0, 1)
        #     t_idx = torch.floor(t_rand_unscaled * num_steps_in_range).long() # 0 ~ 63
            
        #     t = t_idx.float() * d * 2 # t ~ U_discrete[0, 1 - d_big]

        #     t_next = t + d
            
        #     xt_bst = self.scheduler.forward_process(x1_bst, x0_bst, t)

        #     # Get target using two small steps (with EMA model)
        #     with torch.no_grad():
        #         s_t = self.predict(xt_bst, t, d, use_ema=True)
        #         x_t_next = self.scheduler.reverse_process_step(xt_bst, s_t, d=d)
        #         s_t_next = self.predict(x_t_next, t_next, d, use_ema=True)
        #         s_target = (s_t + s_t_next) / 2.0
            
        #     # Predict using one big step (with online model)
        #     s_pred = self.predict(xt_bst, t, d*2, use_ema=False)
            
        #     loss_bst = self.loss_fn(s_pred, s_target)
        #     total_loss += self.bootstrap_ratio * loss_bst
        if B_bst > 0:
            x1_bst, x0_bst = data[:B_bst], noise[:B_bst]

            # [수정] 8개 레벨(0~7)을 샘플링하도록 +1 추가
            # self.log2_M = 7
            dt_base_idx = torch.randint(0, self.log2_M + 1, (B_bst,), device=device) # 0~7 (8개 레벨)
            
            # d_small = {1/2, 1/4, ..., 1/256}
            d = 1.0 / (2*2.0 ** dt_base_idx) 
            
            # [수정] 베이스 케이스(i=7) 처리를 위한 로직
            # d_target은 타겟 생성에 사용할 스텝 크기
            d_target = d.clone()
            
            # 가장 작은 스텝 인덱스 (i == 7, 즉 d_small = 1/256)
            smallest_step_mask = (dt_base_idx == self.log2_M)
            
            # 논문  "When d is at the smallest value... we instead query the model at d=0"
            # d_small이 1/256일 때 (d_big=1/128 학습 시), 타겟 생성에 d=0을 사용
            d_target[smallest_step_mask] = 0.0

            
            # t : [0, 1 - 2d], discrete, multiple of 2d
            # (t 샘플링 로직은 기존 코드와 동일하게 두어도 작동합니다)
            num_steps_in_range = (2.0 ** dt_base_idx).long() 
            t_rand_unscaled = torch.rand(B_bst, device=device) 
            t_idx = torch.floor(t_rand_unscaled * num_steps_in_range).long() 
            
            # d*2 = {1, 1/2, ..., 1/128} (학습할 d_big)
            d_big = d * 2
            t = t_idx.float() * d_big # t ~ U_discrete[0, 1 - d_big]

            xt_bst = self.scheduler.forward_process(x1_bst, x0_bst, t)

            # Get target using two small steps (with EMA model)
            with torch.no_grad():
                # [수정] d 대신 d_target 사용 (i=7일 때 d=0이 됨)
                s_t = self.predict(xt_bst, t, d_target, use_ema=True)
                
                # [수정] d 대신 d_target 사용
                x_t_next = self.scheduler.reverse_process_step(xt_bst, s_t, d=d_target)
                
                # [수정] d 대신 d_target 사용
                t_next = t + d_target
                
                # [수정] d 대신 d_target 사용
                s_t_next = self.predict(x_t_next, t_next, d_target, use_ema=True)
                
                # (d_target=0 이면 s_t == s_t_next 이므로 s_target = s_t 가 됨)
                s_target = (s_t + s_t_next) / 2.0
            
            # Predict using one big step (with online model)
            # [수정] d*2 대신 d_big 사용 (가독성)
            s_pred = self.predict(xt_bst, t, d_big, use_ema=False)
            
            # (i=7일 때) loss = || s(d=1/128) - s_target(from d=0) ||^2
            loss_bst = self.loss_fn(s_pred, s_target)
            total_loss += self.bootstrap_ratio * loss_bst
            
        return total_loss
    
    def predict(self, xt, t, d, use_ema=False, **kwargs):
        """
        Make prediction given noisy data and timestep.
        
        Args:
            xt: Noisy data
            t: Timestep
            **kwargs: Additional arguments (e.g., condition for additional timestep)
            
        Returns:
            Model prediction
        """

        # Ensure sigma is a tensor and on the correct device/shape
        active_network = self.ema_network if use_ema else self.network
        return active_network(xt, t, condition=d)


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

        device = next(self.network.parameters()).device
        
        M = num_inference_timesteps
        d = 1.0 / M
        d_vec = torch.full((shape[0],), d, device=device)
        
        # x_0 ~ N(0, I)
        x = torch.randn(shape, device=device)
        
        traj = [x.cpu()] if return_traj else None

        iterator = tqdm(range(M), desc="Shortcut Sampling", disable=not verbose)
        for i in iterator:
            t = i * d
            t_vec = torch.full((shape[0],), t, device=device)
            
            # s_θ(x_t, t, d)
            v_pred = self.predict(x, t_vec, d_vec, use_ema=True)
            
            # x_{t+d} = x_t + v_pred * d
            x = self.scheduler.reverse_process_step(x, v_pred, d=d)
            
            if return_traj:
                traj.append(x.cpu())
                
        return torch.stack(traj) if return_traj else x


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
    
    kwargs.setdefault('ema_decay_rate', 0.999)
    # Create your model
    model = CustomGenerativeModel(network, scheduler, **kwargs)
    
    return model.to(device)