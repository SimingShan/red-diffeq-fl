from datetime import datetime
import os
import glob
import pickle
import tempfile
from typing import Dict, List, Optional, Tuple
import warnings
import yaml
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import flwr as fl
from flwr.common import (
    Context,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import torch.nn.functional as F
from flwr.server.strategy import FedAvg, FedAvgM, FedOpt, FedProx
from configs.config_utils import AppConfig, load_config
from scripts.data_utils.data_trans import (
    s_denormalize,
    s_normalize,
    s_normalize_none,
    v_denormalize,
    v_normalize,
)
import scripts.data_utils.data_trans as data_trans
import scripts.data_utils.pytorch_ssim  # module has side effects / functions
from scripts.diffusion_models.diffusion_model import *  # TODO: make explicit
from scripts.flwr.flwr_client import *                 # TODO: make explicit
from scripts.flwr.flwr_evaluation import get_evaluate_fn
from scripts.flwr.flwr_utils import *                  # TODO: make explicit
from scripts.pde_solvers.solver import FWIForward

def get_fit_config_fn(config):
    """Factory function to create the on_fit_config_fn."""
    def fit_config_fn(server_round: int):
        # This inner function is what Flower will call
        return {
            "server_round": server_round,
            "local_epochs": config.federated.local_epochs,
            "local_lr": config.federated.local_lr,
            "total_rounds": config.federated.num_rounds,
            "regularization": config.experiment.regularization,
            "reg_lambda": config.experiment.reg_lambda
        }
    return fit_config_fn


def fit_metrics_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics."""
    aggregated = {}
    
    # Store deserialized detailed data separately
    detailed_data_store = []

    for client_id, client_metrics in metrics:
        for key, value in client_metrics.items():
            if key == "detailed_data":
                # Deserialize the data and store it
                deserialized = pickle.loads(value)
                detailed_data_store.append({"client_id": client_id, "data": deserialized})
                continue
            
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(value)
    
    # Calculate means and stds for simple numeric metrics
    final_metrics = {}
    for key, values in aggregated.items():
        if isinstance(values[0], (int, float)):
            final_metrics[f"{key}_mean"] = sum(values) / len(values)
            final_metrics[f"{key}_std"] = np.std(values)
        else:
            final_metrics[key] = values # Should not happen with new client logic
            
    # Add the rich, detailed data to the final metrics dictionary
    if detailed_data_store:
        final_metrics["detailed_client_data"] = detailed_data_store
    
    return final_metrics


def run_full_experiment(config_path: str, process_id: int, run_name: str):
    """
    Loads configuration, runs the full suite of FL experiments for all data families
    and instances, aggregates results, and saves them to a single file.
    """
    assert process_id is not None, "process_id is required"
    assert process_id in [1,2], "process_id must be 1 (CF, CV) or 2 (FF, FV)"
    assert config_path is not None, "config_path is required"
    assert run_name in ['main', 'tuning'], "run_name must be 'main' or 'tuning'"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(config_path)
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- STARTING FULL EXPERIMENT RUN ---")
    print(f"Strategy: {config.experiment.strategy}, Regularization: {config.experiment.regularization}, Scenario: {config.experiment.scenario_flag}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include reg_lambda in directory name for hyperparameter tuning to separate different lambda values
    if run_name == "tuning":
        base_dir_name = f"{run_name}_{config.experiment.strategy}_{config.experiment.regularization}_{config.experiment.reg_lambda}_{config.experiment.scenario_flag}"
    else:
        base_dir_name = f"{run_name}_{config.experiment.strategy}_{config.experiment.regularization}_{config.experiment.scenario_flag}"
    
    # Check for existing directories
    existing_dirs = glob.glob(os.path.join(config.path.output_path, f"{base_dir_name}_*"))
    if existing_dirs:
        # Use the most recent existing directory
        main_output_dir = max(existing_dirs, key=os.path.getctime)
        print(f"Using existing directory: {main_output_dir}")
    else:
        # Create new directory
        main_output_dir = os.path.join(config.path.output_path, f"{base_dir_name}_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        print(f"Created new directory: {main_output_dir}")

    # --- 1. SETUP SHARED COMPONENTS ---
    # These components are the same for all 40 runs.
    # Forward Solver
    ctx = {'n_grid': config.forward.n_grid, 'nt': config.forward.nt, 'dx': config.forward.dx, 
           'nbc': config.forward.nbc, 'dt': config.forward.dt, 'f': config.forward.f,
           'sz': config.forward.sz, 'gz': config.forward.gz, 'ng': config.forward.ng}
    fwi_forward = FWIForward(ctx, device, normalize=True, v_denorm_func=v_denormalize, s_norm_func=s_normalize_none)
    server_diffusion_model = None
    diffusion_state_dict = None
    diffusion_args = None
    diffusion_state_dict = None
    if config.experiment.regularization == "Diffusion":
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
            #unet_model = torch.compile(unet_model, mode="max-autotune")
            diffusion = GaussianDiffusion(
                unet_model,
                image_size=diffusion_args.get('image_size'),
                timesteps=diffusion_args.get('timesteps'),
                sampling_timesteps=diffusion_args.get('sampling_timesteps'),
                objective=diffusion_args.get('objective')
            ).to(device)

            # Load the pretrained weights for the diffusion model
            checkpoint = torch.load(config.path.model_path, map_location=device, weights_only=True)
            state_dict = checkpoint.get('model', checkpoint)
            diffusion.load_state_dict(state_dict)
            diffusion.eval()

            # Assign the created model and its state dict to the variables
            server_diffusion_model = diffusion
            diffusion_state_dict = diffusion.state_dict()

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    
    # This is a good practice: passing config is not redundant, it's dependency injection.
    fit_config_fn = get_fit_config_fn(config)

    # --- 2. SETUP EXPERIMENT LOOP ---
    if process_id == 1:
        families = ['CF', 'CV']
    elif process_id == 2:
        families = ['FF', 'FV']
    
    all_results = {'individual_runs': []}
    family_to_vm = {fam: np.load(f"{config.path.velocity_data_path}/{fam}.npy", mmap_mode="r") for fam in families}
    family_to_gt = {fam: np.load(f"{config.path.gt_seismic_data_path}/{fam}.npy", mmap_mode="r") for fam in families}
    family_to_clients = {
        fam: [np.load(f"{config.path.client_seismic_data_path}/client{c+1}/{fam}.npy", mmap_mode="r")
            for c in range(config.experiment.num_clients)]
        for fam in families
}

    for family in families:
        print(f"\n--- Starting Family: {family} ---")
        
        # Loop over 10 instances in the family
        for i in range(10):
            print(f"-- Running Instance: {family}{i} --")
            
            # --- 3. RE-INITIALIZE FOR EACH RUN ---
            # This is critical to ensure each of the 40 runs is independent.
            
            # Load data for this specific instance
            vm_np = family_to_vm[family][i:i+1, :]
            gt_np = family_to_gt[family][i:i+1, :]
            client_nps = [arr[i:i+1, :] for arr in family_to_clients[family]]

            vm_data = torch.from_numpy(np.ascontiguousarray(vm_np)).float()                     # CPU
            gt_seismic_data = torch.from_numpy(np.ascontiguousarray(gt_np)).float().pin_memory().to(device, non_blocking=True)
            client_data_list = [torch.from_numpy(np.ascontiguousarray(x)).float().pin_memory().to(device, non_blocking=True)
                                for x in client_nps]

            # Create a fresh initial model
            initial_model = data_trans.prepare_initial_model(vm_data, initial_type='smoothed', sigma=config.forward.initial_sigma)
            initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)

            # Create a fresh store for the final model of this run
            final_parameters_store = {}

            # Create a fresh evaluation function with the correct data for this run
            evaluate_fn = get_evaluate_fn(
                model_shape=initial_model.shape, seismic_data=gt_seismic_data, mu_true=vm_data,
                fwi_forward=fwi_forward, data_trans=data_trans, ssim_loss=ssim_loss, device=device,
                total_rounds=config.federated.num_rounds,
                final_params_store=final_parameters_store, config=config,
                diffusion_model=server_diffusion_model
            )
            
            # Create a fresh strategy instance with the new initial model
            strategy_class = {"FedAvg": FedAvg, "FedAvgM": FedAvgM, "FedProx": FedProx}[config.experiment.strategy]
            
            # Base strategy parameters
            strategy_params = {
                "fraction_fit": 1.0, 
                "min_fit_clients": config.experiment.num_clients,
                "min_available_clients": config.experiment.num_clients, 
                "evaluate_fn": evaluate_fn,
                "fraction_evaluate": 0.0, 
                "on_fit_config_fn": fit_config_fn,
                "initial_parameters": ndarrays_to_parameters(tensor_to_ndarrays(initial_model.to(device))),
            }
            
            # Add strategy-specific parameters
            if config.experiment.strategy == "FedAvgM":
                strategy_params["server_momentum"] = config.experiment.server_momentum
            elif config.experiment.strategy == "FedProx":
                # FedProx might need additional parameters
                pass
            
            # Add fit_metrics_aggregation_fn for client metrics collection (both FedAvg and FedAvgM support it)
            strategy_params["fit_metrics_aggregation_fn"] = fit_metrics_fn
            
            strategy = strategy_class(**strategy_params)
            
            # Create a fresh client function instance with the correct data partitions
            client_fn_instance = client_fn_factory(
                partitions=client_data_list, config=config, device=device,
                fwi_forward=fwi_forward, data_trans=data_trans, ssim_loss=ssim_loss,
                diffusion_args=diffusion_args,
                diffusion_state_dict=diffusion_state_dict
            )
            ray_temp_dir = "/tmp/rfl" 
            os.makedirs(ray_temp_dir, exist_ok=True)

            # Cap CPU threads to avoid oversubscription
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1)
            # --- 4. RUN ONE SIMULATION ---
            history = fl.simulation.start_simulation(
                client_fn=client_fn_instance,
                num_clients=config.experiment.num_clients,
                config=fl.server.ServerConfig(num_rounds=config.federated.num_rounds),
                strategy=strategy,
                client_resources={"num_gpus": config.resources.num_gpus_per_client, 
                 "num_cpus": config.resources.num_cpus_per_client} if device.type == "cuda" else {},
                ray_init_args={"include_dashboard": False, "_temp_dir": ray_temp_dir},
            )

            # --- 5. STORE INDIVIDUAL RESULTS ---
            saved_ndarrays = final_parameters_store["final_model"]
            final_model = ndarrays_to_tensor(saved_ndarrays, device)
            
            run_result = {
            "family": family,
            "instance": i,
            "final_model": final_model.cpu().detach().numpy(),
            "all_round_models": final_parameters_store,  # All models from all rounds
            "history": history,  # Server-side metrics
            "config": config
            }

            all_results['individual_runs'].append(run_result)

            result_filename = os.path.join(main_output_dir, f"{family}_{i}_result.pkl")

            with open(result_filename, 'wb') as f:
                pickle.dump(run_result, f)
                    
        print(f"--- Finished Family: {family} ---")

    # --- 6. FINAL SUMMARY ---
    print(f"\n--- EXPERIMENT COMPLETE ---")
    print(f"Process {process_id} completed: {len(families)} families, {len(families) * 10} instances")
    print(f"Individual results saved in: {main_output_dir}")

    return all_results