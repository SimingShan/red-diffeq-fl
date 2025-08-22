import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from .flwr_utils import *
from .regularizations import *

def get_evaluate_fn(
    model_shape: tuple,
    seismic_data: torch.Tensor,
    mu_true: torch.Tensor,
    fwi_forward,
    data_trans,
    ssim_loss,
    device: torch.device,
    total_rounds: int,
    final_params_store: dict,
    diffusion_model=None,
    config=None
):

    def evaluate(server_round: int, parameters: NDArrays, config_from_flwr: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        scenario_flag = config.experiment.scenario_flag
        regularization = config.experiment.regularization
        reg_lambda = config.experiment.reg_lambda
        assert config is not None, "Config must be provided for evaluation."

        if server_round == total_rounds:
            print(f"Server evaluation: Capturing final model parameters at round {server_round}.")
            final_params_store["final_model"] = parameters
        elif server_round % 10 == 0:
            final_params_store[f"model_round_{server_round}"] = parameters

        model = ndarrays_to_tensor(parameters, device)
        with torch.no_grad():
            model_input = model[:, :, 1:-1, 1:-1]
            predicted_seismic = fwi_forward(model_input, scenario=scenario_flag, client_idx=None, num_clients=None)
            seismic_data_dev = seismic_data.to(device)
            seismic_loss = l1_loss_fn(seismic_data_dev.float(), predicted_seismic.float())
            diffusion_loss = torch.tensor(0.0, device=device)
            if diffusion_model is not None and regularization == 'Diffusion':
                batch_size = seismic_data_dev.shape[0]
                time = torch.randint(0, diffusion_model.num_timesteps, (1,)).item()
                time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
                time_tensor = torch.full((batch_size,), time, device=device, dtype=torch.long)
                sigma_x0 = 0.0001
                noise_x0 = torch.randn_like(model)
                x0_pred = model + sigma_x0 * noise_x0
                noise = torch.randn_like(model)
                x_t = diffusion_model.q_sample(x0_pred, t=time_tensor, noise=noise)
                self_cond = x0_pred if diffusion_model.self_condition else None
                model_predictions = diffusion_model.model_predictions(
                    x_t, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
                )
                pred_noise, _ = model_predictions.pred_noise, model_predictions.pred_x_start
                et = pred_noise
                diffusion_loss = torch.mul((et - noise), model).mean()
                reg_loss = diffusion_loss
                total_loss = reg_lambda * diffusion_loss + seismic_loss 
            elif regularization == 'Total_Variation':
                reg_loss = total_variation_loss(model_input)
                total_loss = seismic_loss + reg_lambda * reg_loss
            elif regularization == 'Tiknov':
                reg_loss = tikhonov_loss(model_input)
                total_loss = seismic_loss + reg_lambda * reg_loss
            elif regularization == 'None':
                total_loss = seismic_loss
                reg_loss = torch.tensor(0.0, device=device)
                
            vm_sample = model_input.detach().to('cpu')
            vm_true_norm = data_trans.v_normalize(mu_true)
            if vm_true_norm.dim() == 2:
                vm_true_norm = vm_true_norm.unsqueeze(0).unsqueeze(0)
            vm_true_norm = vm_true_norm.to('cpu')

            if vm_sample.shape != vm_true_norm.shape:
                h_diff = vm_sample.shape[2] - vm_true_norm.shape[2]
                w_diff = vm_sample.shape[3] - vm_true_norm.shape[3]
                h_start, w_start = h_diff // 2, w_diff // 2
                h_end, w_end = h_start + vm_true_norm.shape[2], w_start + vm_true_norm.shape[3]
                if h_diff >= 0 and w_diff >= 0 and h_diff % 2 == 0 and w_diff % 2 == 0:
                    vm_sample = vm_sample[:, :, h_start:h_end, w_start:w_end]
                else:
                    print(f"Warning: Cannot reconcile shapes for metric calculation: {vm_sample.shape} vs {vm_true_norm.shape}")
                    return total_loss.item(), {
                        "seismic_loss": seismic_loss.item(),
                        "diffusion_loss": diffusion_loss.item(),
                        "mae": -1.0, "rmse": -1.0, "ssim": -1.0
                    }

            mae = l1_loss_fn(vm_sample, vm_true_norm)
            mse = l2_loss_fn(vm_sample, vm_true_norm)
            rmse = np.sqrt(mse.item())
            ssim_val = ssim_loss((vm_sample + 1) / 2, (vm_true_norm + 1) / 2)

        return total_loss.item(), {
            'seismic_loss': seismic_loss.item(),
            'reg_loss': reg_loss.item(),
            'mae': mae.item(), 'rmse': rmse, 'ssim': ssim_val.item()
        }

    evaluate.__annotations__['parameters'] = NDArrays
    return evaluate
