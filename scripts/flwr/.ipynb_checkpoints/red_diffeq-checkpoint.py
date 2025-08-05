import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from functools import partial

# Import these from your existing modules
from . import data_trans
from . import data_vis
from . import pytorch_ssim 
# Define the loss functions 
Huber = nn.SmoothL1Loss()
l1 = nn.L1Loss()
l2 = nn.MSELoss()
ssim_loss = pytorch_ssim.SSIM(window_size=11)  

def total_variation_loss(mu):
    """Computes Total Variation Loss for the velocity map (mu)."""
    diff_x = torch.abs(mu[:, :, :, 1:] - mu[:, :, :, :-1])  # Horizontal differences
    diff_y = torch.abs(mu[:, :, 1:, :] - mu[:, :, :-1, :])  # Vertical differences
    tv_loss = torch.mean(diff_x) + torch.mean(diff_y)
    return tv_loss

def tikhonov_loss(mu):
    """Computes L2 regularization (Tikhonov) for the velocity map (mu)."""
    diff_x = mu[:, :, :, 1:] - mu[:, :, :, :-1]
    diff_y = mu[:, :, 1:, :] - mu[:, :, :-1, :]
    
    # Compute the L2 loss separately for x and y directions
    l2_loss_x = torch.mean(diff_x ** 2)
    l2_loss_y = torch.mean(diff_y ** 2)
    
    # Sum the losses from both directions
    l2_loss = l2_loss_x + l2_loss_y
    
    return l2_loss

class RED_DiffEq:
    """
    REgularization by Denoising Diffusion (RED-Diff) algorithm implementation.
    This class implements the RED-Diff algorithm for inverse problems using 
    pre-trained diffusion models as regularizers.
    """
    
    def __init__(self, diffusion_model, data_trans_module, data_vis_module, pytorch_ssim_module):
        """
        Initialize RED-Diff with a pre-trained diffusion model.
        
        Args:
            diffusion_model: The pre-trained diffusion model (GaussianDiffusion instance)
            data_trans_module: Module containing data transformation functions
            data_vis_module: Module containing visualization functions
            pytorch_ssim_module: Module containing SSIM implementation
        """
        self.diffusion_model = diffusion_model
        self.data_trans = data_trans_module
        self.data_vis = data_vis_module
        self.ssim_loss = pytorch_ssim_module.SSIM(window_size=11)
        self.device = diffusion_model.device
    
    def sample(self, mu, mu_true, y, ts, lr=0.001, reg_lambda=0.01, fwi_forward=None, plot_show=False, 
              loss_type="l2", noise_std=0, missing_number=0, regularization=None):
        """
        Main RED-Diff sampling function.
        
        Args:
            mu: Initial velocity map (tensor)
            mu_true: Ground truth velocity map (tensor) - for evaluation only
            y: Observed seismic data (tensor)
            ts: Number of optimization steps
            lr: Learning rate for optimization
            fwi_forward: The forward modeling function
            plot_show: Whether to show plots during optimization
            loss_type: Type of loss function ("l1", "l2", or "Huber")
            noise_std: Standard deviation of noise to add to seismic data
            missing_number: Number of missing traces to simulate
            regularization: Type of regularization ("diffusion", "l2", "tv", "hybrid", or None)
            
        Returns:
            Tuple containing optimized velocity map and various loss metrics
        """
        # Set initial assertions
        assert mu.shape[0] == y.shape[0], "The batch size of the velocity map has to equal the seismic data"
        assert loss_type in ["l1", "l2", "Huber"], "Please choose the loss function from 'l1', 'l2', or 'Huber'"
        assert regularization in ["diffusion", "l2", "tv", "hybrid", None], "Please choose the regularization from 'diffusion', 'l2', 'tv', or None"
    
        # Ensure tensors are float32
        mu = mu.float()
        mu_true = mu_true.float()
        y = torch.tensor(y).float().to(self.device)
    
        batch_size = mu.shape[0]
        device = self.device
    
        # Optimizer setup with Cosine Annealing
        mu = torch.autograd.Variable(mu, requires_grad=True)
        optimizer = torch.optim.Adam([mu], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ts, eta_min=0.0)
    
        # Initialize lists to store loss values
        total_losses = []
        noise_losses = []
        obs_losses = []
        reg_losses = []
        reg_losses_raw = []
        learning_rates = []
        current_MAE = []
        current_RMSE = []
        current_SSIM = []
    
        # Apply noise and missing traces to seismic data
        y = self.data_trans.add_noise_to_seismic(y, noise_std)
        y = self.data_trans.missing_trace(y, missing_number)
    
        if plot_show:
            self.data_vis.plot_single_seismic_2(y[0, 2, :, :])
        y = torch.tensor(y).to(device)
    
        # Initialize the tqdm progress bar
        pbar = tqdm(range(ts), desc="Optimizing", unit="step")
    
        for l in pbar:
            if regularization in ['diffusion', 'hybrid']:
                # Randomly draw time step
                time = torch.randint(0, self.diffusion_model.num_timesteps, (1,)).item()
                time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
    
                # Generate noisy predictions
                time_tensor = torch.full((batch_size,), time, device=device, dtype=torch.long)
                sigma_x0 = 0.0001
                noise_x0 = torch.randn_like(mu)
                x0_pred = mu + sigma_x0 * noise_x0
                noise = torch.randn_like(mu)    
                x_t = self.diffusion_model.q_sample(x0_pred, t=time_tensor, noise=noise)
                self_cond = x0_pred if self.diffusion_model.self_condition else None
                model_predictions = self.diffusion_model.model_predictions(
                    x_t, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
                )
                pred_noise, x_start = model_predictions.pred_noise, model_predictions.pred_x_start
                et = pred_noise.detach()
                raw_loss_noise = torch.mul((et - noise).detach(), mu).mean()
            else:
                x0_pred = mu
                loss_noise = torch.tensor([0], device=device).float()
                raw_loss_noise = torch.tensor([0], device=device).float()
    
            # Seismic forward modeling for loss_obs
            predicted_seismic = fwi_forward(x0_pred[:, :, 1:-1, 1:-1])  
    
            if loss_type == 'l1':
                loss_obs = l1(y.float(), predicted_seismic.float())
            elif loss_type == 'l2':
                loss_obs = l2(y.float(), predicted_seismic.float())
            elif loss_type == 'Huber':
                loss_obs = Huber(y.float(), predicted_seismic.float())
    
            # Apply regularization based on type
            if regularization == 'tv':
                raw_reg_loss = total_variation_loss(x0_pred)
                reg_loss = reg_lambda * raw_reg_loss
            elif regularization == 'l2':
                raw_reg_loss = tikhonov_loss(x0_pred)
                reg_loss = reg_lambda * raw_reg_loss
            elif regularization == 'diffusion':
                raw_reg_loss = raw_loss_noise
                reg_loss = reg_lambda * raw_reg_loss
            else:
                raw_reg_loss = torch.tensor([0], device=device).float()
                reg_loss = torch.tensor([0], device=device).float()
    
            loss = loss_obs + reg_loss
    
            # Store loss values
            total_losses.append(loss.item())
            noise_losses.append(raw_loss_noise.item())
            obs_losses.append(loss_obs.item())
            reg_losses.append(reg_loss.item())
            reg_losses_raw.append(raw_reg_loss.item())
            learning_rates.append(scheduler.get_last_lr()[0])
    
            # Compute additional evaluation metrics
            vm_sample_unnorm = x0_pred[:, :, 1:-1, 1:-1].detach().to('cpu')
            vm_data_unnorm = self.data_trans.v_normalize(mu_true)
            mae = l1(vm_sample_unnorm, vm_data_unnorm)
            mse = l2(vm_sample_unnorm, vm_data_unnorm)
            rmse = np.sqrt(mse.item())
            ssim = self.ssim_loss((vm_sample_unnorm + 1) / 2, (vm_data_unnorm + 1) / 2)
            current_MAE.append(mae.item())
            current_RMSE.append(rmse)
            current_SSIM.append(ssim.item())
    
            if plot_show:
                self.data_vis.plot_single_v(self.data_trans.v_denormalize(mu))
    
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mu.data.clamp_(-1, 1)
            scheduler.step()
    
            # Update progress bar
            pbar.set_postfix({
                'total_loss': loss.item(),
                'obs_loss': loss_obs.item(),
                'reg_loss': reg_loss.item(),
                'MAE': mae.item(),
                'SSIM': ssim.item()
            })
    
        return mu, total_losses, obs_losses, reg_losses, reg_losses_raw, current_MAE, current_RMSE, current_SSIM

