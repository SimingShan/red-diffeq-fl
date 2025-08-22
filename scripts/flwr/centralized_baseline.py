import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.pyplot as plt

# Define the loss functions 
l1 = nn.L1Loss()
l2 = nn.MSELoss()

def scenario_aware_seismic_loss(y, predicted_seismic, scenario):
    """
    Compute L1 loss only on source–receiver blocks that exist in the federated setting.
    The mask is built as a UNION of per-client coverage.
    Shapes: (B, num_sources, T, num_receivers)
    """
    if y.shape != predicted_seismic.shape:
        raise ValueError(f"Shape mismatch: y {y.shape} vs pred {predicted_seismic.shape}")

    device = y.device
    mask = torch.zeros_like(y, dtype=y.dtype, device=device)

    # Build union-of-coverage mask per scenario
    if scenario == '2A':
        mask[:, :3,  :, :35] = 1  # client 0
        mask[:, 3:5, :, 35:] = 1  # client 1
    elif scenario == '2B':
        mask[:, :3,  :, :35] = 1  # client 0
        mask[:, 2:5, :, 35:] = 1  # client 1 (overlaps source 2)
    elif scenario == '2C':
        mask[:, :3,  :, :35] = 1  # client 0
        mask[:, 3:6, :, 35:] = 1  # client 1
    elif scenario == '3A':
        mask[:, :4,  :, :24]  = 1  # client 0
        mask[:, 3:7, :, 24:47] = 1  # client 1 (overlaps 3,6)
        mask[:, 6:10, :, 47:] = 1  # client 2
    elif scenario == '3B':
        mask[:, 0:3, :, :24]   = 1  # client 0
        mask[:, 3:7, :, 24:47] = 1  # client 1
        mask[:, 7:10, :, 47:]   = 1  # client 2
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    # Consistent scaling across scenarios
    diff = (y - predicted_seismic).abs() * mask
    denom = mask.sum().clamp_min(1.0)  # number of active elements

    return diff.sum() / denom



class RED_DiffEq:
    
    def __init__(self, diffusion_model, data_trans_module, pytorch_ssim_module):

        self.diffusion_model = diffusion_model
        self.data_trans = data_trans_module
        self.ssim_loss = pytorch_ssim_module.SSIM(window_size=11)
        self.device = diffusion_model.device
    
    def sample(self, mu, mu_true, y, ts, lr=0.001, reg_lambda=0.01, fwi_forward=None, scenario=None):
        assert scenario is not None and scenario in ['2A', '2B', '2C', '3A', '3B'], "Scenario not supported"
        
        result = {'total_losses': [],
                  'noise_losses': [],
                  'obs_losses': [],
                  'reg_losses': [],
                  'learning_rates': [],
                  'mae': [],
                  'rmse': [],
                  'ssim': [],
                  'final_model': []}

        mu = mu.float()
        mu_true = mu_true.float()
        y = torch.tensor(y).float().to(self.device)

        batch_size = mu.shape[0]
        device = self.device
    
        # Optimizer setup with Cosine Annealing
        mu = torch.autograd.Variable(mu, requires_grad=True)
        optimizer = torch.optim.Adam([mu], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ts, eta_min=0.0)
    
        # Initialize the tqdm progress bar
        pbar = tqdm(range(ts), desc="Optimizing", unit="step")
    
        for l in pbar:
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
            diffusion_loss = torch.mul((et - noise).detach(), x0_pred).mean()

            predicted_seismic = fwi_forward(x0_pred[:, :, 1:-1, 1:-1])  
    
            # Use scenario-aware loss that mimics federated data partitioning
            # This ensures fair comparison by only computing loss on areas where data exists
            loss_obs = scenario_aware_seismic_loss(y, predicted_seismic, scenario)
            reg_loss = diffusion_loss
            total_loss = loss_obs + reg_lambda * reg_loss
    
            # Store loss values
            result['total_losses'].append(total_loss.item())
            result['noise_losses'].append(diffusion_loss.item())
            result['obs_losses'].append(loss_obs.item())
            result['reg_losses'].append(reg_loss.item())
            result['total_losses'].append(total_loss.item())
            result['learning_rates'].append(scheduler.get_last_lr()[0])
    
            # Compute additional evaluation metrics
            vm_sample_unnorm = x0_pred[:, :, 1:-1, 1:-1].detach().to('cpu')
            vm_data_unnorm = self.data_trans.v_normalize(mu_true).to('cpu')
            mae = l1(vm_sample_unnorm, vm_data_unnorm)
            mse = l2(vm_sample_unnorm, vm_data_unnorm)
            rmse = np.sqrt(mse.item())
            ssim = self.ssim_loss((vm_sample_unnorm + 1) / 2, (vm_data_unnorm + 1) / 2)
            result['mae'].append(mae.item())
            result['rmse'].append(rmse)
            result['ssim'].append(ssim.item())

            # Update parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            mu.data.clamp_(-1, 1)
            scheduler.step()

            # Update progress bar
            pbar.set_postfix({
                'total_loss': total_loss.item(),
                'obs_loss': loss_obs.item(),
                'reg_loss': reg_loss.item(),
                'MAE': mae.item(),
                'SSIM': ssim.item()
            })
        result['final_model'] = mu.detach().to('cpu').numpy()
    
        return result

