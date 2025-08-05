import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from functools import partial
import copy
import time
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import yaml
import os
import tempfile
import pickle
from datetime import datetime
from flwr.server.strategy import FedAvg, FedProx, FedAvgM, FedOpt
from scripts.diffusion_models.diffusion_model import *
from scripts.pde_solvers.pde_solver_5clients import FWIForward 
from torch.optim import Adam
from scripts.data_utils.data_trans import v_normalize, v_denormalize, s_normalize_none, s_normalize, s_denormalize
import torch.multiprocessing as mp
from configs.config_utils import AppConfig

### Load configuration & setup diffusion model ###

def load_config(path):
    with open(path) as f:
        raw_config = yaml.safe_load(f)
    return AppConfig(**raw_config)

config = load_config('configs/config_5clients.yml')

mp.set_start_method('spawn', force=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## setup forward solver ##
ctx = {
    'n_grid': config.forward.n_grid, 'nt': config.forward.nt, 
    'dx': config.forward.dx, 'nbc': config.forward.nbc, 
    'dt': config.forward.dt, 'f': config.forward.f,
    'sz': config.forward.sz, 'gz': config.forward.gz,
    'ng': config.forward.ng, 'ns': config.forward.ns
}

fwi_forward = FWIForward(ctx, device, normalize=True, 
                         v_denorm_func=v_denormalize, 
                         s_norm_func=s_normalize_none)

## setup diffusion model ##
diffusion_args = {
    'dim': config.diffusion.dim, 'dim_mults': config.diffusion.dim_mults, 
    'flash_attn': config.diffusion.flash_attn, 'channels': config.diffusion.channels,
    'image_size': config.diffusion.image_size, 'timesteps': config.diffusion.timesteps, 
    'sampling_timesteps': config.diffusion.sampling_timesteps, 
    'objective': config.diffusion.objective
}


unet_model = Unet(
    dim=diffusion_args.get('dim'),
    dim_mults=diffusion_args.get('dim_mults'),
    flash_attn=diffusion_args.get('flash_attn'),
    channels=diffusion_args.get('channels')
)

diffusion = GaussianDiffusion(
    unet_model,
    image_size=diffusion_args.get('image_size'),
    timesteps=diffusion_args.get('timesteps'),
    sampling_timesteps=diffusion_args.get('sampling_timesteps'),
    objective=diffusion_args.get('objective')
).to(device)

### Load & prepare data ###
velocity_data_path = config.path.velocity_data_path
seismic_data_path = config.path.seismic_data_path
model_path = config.path.model_path
output_path = config.path.output_path

checkpoint = torch.load(model_path, weights_only=True)
state_dict = checkpoint.get('model', checkpoint)
diffusion.load_state_dict(state_dict)
diffusion.eval()
unwrapped_diffusion_model = diffusion
unwrapped_diffusion_model.eval()
diffusion_state_dict = unwrapped_diffusion_model.state_dict()

seismic_data = torch.tensor(np.load(seismic_data_path + "/CF_test.npy")[0:1,:]).float().to(device)
vm_data = torch.tensor(np.load(velocity_data_path + "/CF_test.npy")[0:1,:]).float()

initial_model = data_trans.prepare_initial_model(vm_data, initial_type='smoothed', 
                                                 sigma=config.forward.initial_sigma)
initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)

def client_fn_factory(partitions: List[torch.Tensor]):
    """
    This is a factory function that returns a new `client_fn` for simulation.
    The returned `client_fn` will have the `partitions` in its closure.
    """

    def client_fn(context: Context) -> flwr.client.Client:
        """This is the actual client function that will be executed by the actor."""
        
        # ==================== THE FINAL FIX ====================
        # Ignore the broken context.node_id.
        # Get the correct ID from the node_config dictionary and convert to int.
        client_id = int(context.node_config["partition-id"])
        # =======================================================
        
        # This line will now work correctly with the proper client_id.
        local_data_for_client = partitions[client_id]

        # Instantiate your FwiClient as before
        fwi_client_instance = FwiClient(
            cid=str(client_id),
            device=device,
            fwi_forward=fwi_forward,
            data_trans=data_trans,
            ssim_loss=ssim_loss,
            local_data=local_data_for_client,
            num_total_clients=config.experiment.num_clients,
            diffusion_state_dict=diffusion_state_dict,
            diffusion_model_structure_args=diffusion_args
        )
        
        return fwi_client_instance.to_client()

    # Return the configured client_fn
    return client_fn

from scripts.flwr.flwr_client import FwiClient
from scripts.flwr.flwr_evaluation import get_evaluate_fn
from scripts.flwr.flwr_utils import *
import scripts.data_utils.pytorch_ssim 
from flwr.common import Context
import flwr
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
if diffusion_state_dict is not None and diffusion_args is not None:
    server_diffusion_model = diffusion
    
ssim_loss = pytorch_ssim.SSIM(window_size=11)  
final_parameters_store = {} 
num_clients = config.experiment.num_clients
regularization = config.experiment.regularization
fed_rounds = config.federated.num_rounds
local_epochs = config.federated.local_epochs
local_lr = config.federated.local_lr

evaluate_fn = get_evaluate_fn(
    model_shape=initial_model.shape,
    seismic_data=seismic_data,
    mu_true=vm_data,
    fwi_forward=fwi_forward,
    data_trans=data_trans,
    ssim_loss=ssim_loss,
    device=device,
    diffusion_model=server_diffusion_model,
    total_rounds=config.federated.num_rounds,
    final_params_store=final_parameters_store
)

def fit_config_fn(server_round: int):
    return {
        "server_round": server_round, "local_epochs": local_epochs,
        "local_lr": local_lr, "total_rounds": fed_rounds, "regularization": regularization
    }
client_data_partitions = []

for i in range(5):
    client_data_partitions.append(seismic_data[:,i:i+1,:,:])
    
for i, data in enumerate(client_data_partitions):
    print(f"Data shape for Client {i}: {data.shape}")


client_fn_instance = client_fn_factory(client_data_partitions)

strategy_classes = {
    "FedAvg": FedAvg,
    "FedProx": FedProx, 
    "FedAvgM": FedAvgM,
    "FedOpt": FedOpt
}

# Get the actual class
strategy_class = strategy_classes[config.experiment.strategy]
strategy_params = {"server_momentum": config.experiment.server_momentum}

strategy = strategy_class(
    fraction_fit=1.0, 
    min_fit_clients=config.experiment.num_clients,
    min_available_clients=config.experiment.num_clients,
    evaluate_fn=evaluate_fn, 
    fraction_evaluate=0.0, 
    on_fit_config_fn=fit_config_fn,
    initial_parameters=ndarrays_to_parameters(tensor_to_ndarrays(initial_model.to(device))),
    **strategy_params
)
client_resources = {"num_cpus": 1, "num_gpus": 0.5} if device.type == 'cuda' else {"num_cpus": 2}
ray_init_args = {"include_dashboard": False, "_temp_dir": os.path.join(os.getcwd(), "ray_temp")}

history = fl.simulation.start_simulation(
    client_fn=client_fn_instance, # Use the function from the factory
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=fed_rounds),
    strategy=strategy,
    client_resources=client_resources,
    ray_init_args=ray_init_args
)

final_model = None
if "final_model" in final_parameters_store:
    saved_ndarrays = final_parameters_store["final_model"]
    final_model = ndarrays_to_tensor(saved_ndarrays, device)
    print("Successfully loaded final model.")
else:
    print("Warning: final_model not found in store. Using initial model.")
    final_model = test_initial_model

if final_model is None:
    print(f"Warning: Could not retrieve final model for {strategy_name}, using initial model")
    final_model = test_initial_model
with torch.no_grad():
    model_input = final_model[:, :, 1:-1, 1:-1]
    predicted_seismic = fwi_forward(model_input)
    seismic_loss = l1_loss_fn(seismic_data.float(), predicted_seismic.float())
    vm_sample = model_input.detach().to('cpu')
    vm_true_norm = v_normalize(vm_data)
    if vm_true_norm.dim() == 2:
        vm_true_norm = vm_true_norm.unsqueeze(0).unsqueeze(0)
    vm_true_norm = vm_true_norm.to('cpu')

    if vm_sample.shape != vm_true_norm.shape:
        h_diff = vm_sample.shape[2] - vm_true_norm.shape[2]
        w_diff = vm_sample.shape[3] - vm_true_norm.shape[3]
        if h_diff >= 0 and w_diff >= 0 and h_diff % 2 == 0 and w_diff % 2 == 0:
            h_start, w_start = h_diff // 2, w_diff // 2
            h_end, w_end = h_start + vm_true_norm.shape[2], w_start + vm_true_norm.shape[3]
            vm_sample = vm_sample[:, :, h_start:h_end, w_start:w_end]

    mae = l1_loss_fn(vm_sample, vm_true_norm).item()
    mse = l2_loss_fn(vm_sample, vm_true_norm).item()
    rmse = np.sqrt(mse)
    ssim_val = ssim_loss((vm_sample + 1) / 2, (vm_true_norm + 1) / 2).item()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

strategy_results = {
    'strategy': strategy_name,
    'final_model': final_model.cpu().detach().numpy(),
    'metrics_history': history.metrics_centralized,
    'losses_history': history.losses_centralized,
    'final_metrics': {
        'mae': mae,
        'rmse': rmse,
        'ssim': ssim_val,
        'seismic_loss': seismic_loss.item()
    },

    'config': {
        'fed_rounds': fed_rounds,
        'local_epochs': local_epochs,
        'local_lr': local_lr,
        'num_clients': num_clients,
        'strategy_params': strategy_params
    }
}

filename = f"results/{strategy_name}_results_{timestamp}.pkl"
with open(filename, 'wb') as f:
    pickle.dump(strategy_results, f)
print(f"{strategy_name} results saved to {filename}")

with open(f"results/{strategy_name}_summary_{timestamp}.txt", 'w') as f:
    f.write(f"{strategy_name} Results\n")
    f.write(f"Run at: {timestamp}\n\n")
    f.write(f"Final MAE: {mae:.6f}\n")
    f.write(f"Final RMSE: {rmse:.6f}\n")
    f.write(f"Final SSIM: {ssim_val:.6f}\n")
    f.write(f"Final Seismic Loss: {seismic_loss.item():.6f}\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  Fed Rounds: {fed_rounds}\n")
    f.write(f"  Local Epochs: {local_epochs}\n")
    f.write(f"  Local Learning Rate: {local_lr}\n")
    f.write(f"  Number of Clients: {num_clients}\n")
    f.write(f"  Strategy Parameters: {strategy_params}\n")

all_results[strategy_name] = strategy_results
plot_path = f"results/{strategy_name}_plots"
os.makedirs(plot_path, exist_ok=True)