import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from src.regularization_methods.benchmarks import total_variation_loss, tikhonov_loss
from src.regularization_methods.red_diffeq import RED_DiffEq
from src.federated_learning.centralized_loss import scenario_aware_seismic_loss
# Define the loss functions 
Huber = nn.SmoothL1Loss()
l1 = nn.L1Loss()
l2 = nn.MSELoss()

class ResultsDict:
    """Helper to accumulate losses and image-quality metrics over optimization."""
    def __init__(self, data_trans_module, ssim_loss, loss_type, regularization_method, reg_lambda, scenario=None):
        self.data_trans = data_trans_module
        self.ssim_loss = ssim_loss
        self.loss_type = loss_type
        self.results_dict = self.create_results_dict()
        self.regularization_method = regularization_method
        self.scenario = scenario
    def create_results_dict(self):
        results_dict = {
            'total_losses': [],
            'obs_losses': [],
            'reg_losses': [],
            'ssim': [],
            'mae': [],
            'rmse': []
        }
        return results_dict 
    
    def calculate_metrics(self, mu, mu_true, y):
        """Compute MAE, RMSE, and SSIM on the interior region to avoid boundary effects."""
        # Compare on the same interior window for both tensors to avoid shape mismatch
        vm_sample_unnorm = mu[:, :, 1:-1, 1:-1].detach().to('cpu')
        vm_data_unnorm = self.data_trans.v_normalize(mu_true).detach().to('cpu')
        mae = l1(vm_sample_unnorm, vm_data_unnorm)
        mse = l2(vm_sample_unnorm, vm_data_unnorm)
        rmse = np.sqrt(mse.item())
        ssim = self.ssim_loss((vm_sample_unnorm + 1) / 2, (vm_data_unnorm + 1) / 2)
        return mae, rmse, ssim

    def calcualte_seismic_loss(self, predicted_seismic, y, loss_type):
        """Calculate the observation loss between predicted and measured seismic data."""
        if self.scenario is not None:
            loss = scenario_aware_seismic_loss(y, predicted_seismic, self.scenario)
        else:
            if self.loss_type == 'l1':
                loss = l1(y.float(), predicted_seismic.float())
            elif self.loss_type == 'l2':
                loss = l2(y.float(), predicted_seismic.float())
            elif self.loss_type == 'Huber':
                loss = Huber(y.float(), predicted_seismic.float())
        return loss
    
    def calcualte_raw_reg_loss(self, mu, reg_lambda):
        """Compute the raw regularization loss given the current `mu`."""
        raw_reg_loss = self.regularization_method.get_reg_loss(mu)
        return raw_reg_loss
    
    def calcualte_total_loss(self, loss_obs, raw_reg_loss, reg_lambda):
        """Combine observation loss and regularization loss with weight `reg_lambda`."""
        total_loss = loss_obs + reg_lambda * raw_reg_loss
        return total_loss
    
    def update(self, total_losses, obs_losses, reg_losses, ssim, mae, rmse):
        # Convert tensors to plain Python floats to avoid keeping autograd graphs in memory
        def _to_float(val):
            try:
                if isinstance(val, torch.Tensor):
                    return val.detach().float().cpu().item()
            except Exception:
                pass
            try:
                return float(val)
            except Exception:
                return val

        self.results_dict['total_losses'].append(_to_float(total_losses))
        self.results_dict['obs_losses'].append(_to_float(obs_losses))
        self.results_dict['reg_losses'].append(_to_float(reg_losses))
        self.results_dict['ssim'].append(_to_float(ssim))
        self.results_dict['mae'].append(_to_float(mae))
        self.results_dict['rmse'].append(_to_float(rmse))

    def get_results(self):
        return self.results_dict

class Regularization_method:
    """Factory for computing regularization based on a selected type."""
    def __init__(self, regularization_type, diffusion_model):
        self.regularization_type = regularization_type
        self.diffusion_model = diffusion_model
    
    def initialize_regularization(self):
        if self.regularization_type == 'Diffusion':
            regularization = RED_DiffEq(self.diffusion_model)
        elif self.regularization_type == 'Tiknov':
            regularization = tikhonov_loss
        elif self.regularization_type == 'Total_Variation':
            regularization = total_variation_loss
        else:
            regularization = None
        return regularization
        
    def get_reg_loss(self, mu):
        """Compute regularization loss for `mu` using the configured method."""
        if self.regularization_type == 'Diffusion':
            reg_loss = RED_DiffEq(self.diffusion_model).get_reg_loss(mu)
        elif self.regularization_type == 'Tiknov':
            reg_loss = tikhonov_loss(mu)
        elif self.regularization_type == 'Total_Variation':
            reg_loss = total_variation_loss(mu)
        else:
            reg_loss = torch.tensor(0.0, device=mu.device).float()
        return reg_loss

class run_inversion:
    """Run the inversion loop with observation and regularization losses."""
    def __init__(self, diffusion_model, data_trans_module, pytorch_ssim_module, regularization):
        self.diffusion_model = diffusion_model
        self.data_trans = data_trans_module
        self.ssim_loss = pytorch_ssim_module.SSIM(window_size=11)
        self.device = diffusion_model.device
        # Initialize regularization method from constructor argument
        self.regularization_method = Regularization_method(regularization, diffusion_model)

    def sample(self, mu, mu_true, y, ts, lr=0.001, reg_lambda=0.01, fwi_forward=None, 
              loss_type="l2", regularization=None, scenario=None):
        assert mu.shape[0] == y.shape[0], "The batch size of the velocity map has to equal the seismic data"
        assert loss_type in ["l1", "l2", "Huber"], "Please choose the loss function from 'l1', 'l2', or 'Huber'"
        assert regularization in ["Diffusion", "Tiknov", "Total_Variation", None], "Please choose the regularization from 'Diffusion', 'Tiknov', 'Total_Variation', or None"

        if fwi_forward is None or not callable(fwi_forward):
            raise ValueError("fwi_forward must be a callable forward modeling function")
        fwi_forward = fwi_forward.to(self.device)
        # If a regularization override is provided at call time, apply it
        if regularization is not None:
            self.regularization_method = Regularization_method(regularization, self.diffusion_model)

        # Ensure tensors are float32 and on the correct device BEFORE creating optimizer
        mu = mu.float()
        mu_true = mu_true.float()
        # Make `mu` a leaf tensor on the correct device with gradients enabled
        mu = mu.clone().detach().to(self.device).requires_grad_(True)
        # Optimizer setup with Cosine Annealing (track gradients for mu)
        optimizer = torch.optim.Adam([mu], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ts, eta_min=0.0)
        results_dict = ResultsDict(self.data_trans, self.ssim_loss, loss_type, self.regularization_method, reg_lambda, scenario=scenario)
        y = y.to(self.device)
    
        # Initialize the tqdm progress bar
        pbar = tqdm(range(ts), desc="Optimizing", unit="step")
    
        for l in pbar:
            # Seismic forward modeling for loss_obs
            # Model expects interior (no padding). Ensure view is not reassigning storage
            # Prefer scenario-aware forward if supported by the solver
            mu_interior = mu[:, :, 1:-1, 1:-1]
            try:
                predicted_seismic = fwi_forward(mu_interior, scenario=scenario)
            except TypeError:
                # Backward compatibility with solvers that do not accept `scenario`
                predicted_seismic = fwi_forward(mu_interior)
            # Calculate the loss
            loss_obs = results_dict.calcualte_seismic_loss(predicted_seismic, y, loss_type)
            raw_reg_loss = results_dict.calcualte_raw_reg_loss(mu, reg_lambda)
            total_loss = results_dict.calcualte_total_loss(loss_obs, raw_reg_loss, reg_lambda)
            # Calculate the metrics
            mae, rmse, ssim = results_dict.calculate_metrics(mu, mu_true, y)
            # Update the results dictionary
            results_dict.update(total_loss, loss_obs, raw_reg_loss, ssim, mae, rmse)
    
            # Update parameters
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            mu.data.clamp_(-1, 1)
            scheduler.step()

            pbar.set_postfix({
                'total_loss': total_loss.item(),
                'obs_loss': loss_obs.item(),
                'reg_loss': raw_reg_loss.item(),
                'MAE': mae.item(),
                'SSIM': ssim.item(),
                'RMSE': rmse.item()
            })
        final_results = results_dict.get_results()
        return mu, final_results
