import torch
from flwr.common import NDArrays
from src.utils import *
from src.full_waveform_inversion import *
from src.federated_learning.flwr_utils import *

def get_evaluate_fn(
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

    def evaluate(server_round: int, parameters: NDArrays, server_config):
        scenario_flag = config.experiment.scenario_flag
        regularization = config.experiment.regularization
        reg_lambda = config.experiment.reg_lambda
        regularization_method = Regularization_method(regularization, diffusion_model)
        assert config is not None, "Config must be provided for evaluation."

        if server_round == total_rounds:
            print(f"Server evaluation: Capturing final model parameters at round {server_round}.")
            final_params_store["final_model"] = parameters
        elif server_round % 10 == 0:
            final_params_store[f"model_round_{server_round}"] = parameters

        model = ndarrays_to_tensor(parameters, device)
        results_dict = ResultsDict(data_trans, ssim_loss, loss_type = 'l1', regularization_method = regularization_method, reg_lambda = reg_lambda)

        with torch.no_grad():
            model_input = model[:, :, 1:-1, 1:-1]
            predicted_seismic = fwi_forward(model_input, scenario=scenario_flag, client_idx=None, num_clients=None)
            seismic_data_dev = seismic_data.to(device)
            loss_obs = results_dict.calcualte_seismic_loss(predicted_seismic, seismic_data_dev, loss_type = 'l1')
            raw_reg_loss = results_dict.calcualte_raw_reg_loss(model_input, reg_lambda)
            total_loss = results_dict.calcualte_total_loss(loss_obs, raw_reg_loss, reg_lambda)
            mae, rmse, ssim = results_dict.calculate_metrics(model_input, mu_true, seismic_data_dev)
            results_dict.update(total_loss, loss_obs, raw_reg_loss, ssim, mae, rmse)

        return total_loss.item(), {
            'seismic_loss': loss_obs.item(),
            'reg_loss': raw_reg_loss.item(),
            'mae': mae.item(), 'rmse': rmse, 'ssim': ssim.item()
        }

    evaluate.__annotations__['parameters'] = NDArrays
    return evaluate
