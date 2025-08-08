# ========================
# Standard library imports
# ========================
from datetime import datetime
import os
import pickle
import tempfile
from typing import Dict, List, Optional, Tuple
import warnings
import yaml
from omegaconf import OmegaConf
# =========================
# Third-party library imports
# =========================
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import Adam

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
from flwr.server.strategy import FedAvg, FedAvgM, FedOpt, FedProx

# =========================
# Local/project imports
# =========================
from configs.config_utils import AppConfig, load_config

from scripts.data_utils.data_trans import (
    s_denormalize,
    s_normalize,
    s_normalize_none,
    v_denormalize,
    v_normalize,
)
import scripts.data_utils.pytorch_ssim  # module has side effects / functions

from scripts.diffusion_models.diffusion_model import *  # TODO: make explicit
from scripts.flwr.flwr_client import *                 # TODO: make explicit
from scripts.flwr.flwr_evaluation import get_evaluate_fn
from scripts.flwr.flwr_utils import *                  # TODO: make explicit
from scripts.pde_solvers.solver import FWIForward


def average_histories(histories: list):
    """Averages a list of Flower History objects."""
    if not histories:
        return {}

    # Average losses
    avg_loss = np.mean([loss for _, loss in histories[0].losses_centralized], axis=0).tolist()
    
    # Average metrics
    avg_metrics = {}
    # Get all metric keys from the first history's metrics dict
    metric_keys = histories[0].metrics_centralized.keys()
    
    for key in metric_keys:
        # Stack all histories for the current metric key
        stacked_metrics = np.array([h.metrics_centralized[key] for h in histories])
        # Average across histories (axis 0)
        avg_metric = np.mean([value for _, value in stacked_metrics[0]], axis=0).tolist()
        avg_metrics[key] = avg_metric
        
    return {"losses_centralized": avg_loss, "metrics_centralized": avg_metrics}


def get_fit_config_fn(config):
    """Factory function to create the on_fit_config_fn."""
    def fit_config_fn(server_round: int):
        # This inner function is what Flower will call
        return {
            "server_round": server_round,
            "local_epochs": config.federated.local_epochs,
            "local_lr": config.federated.local_lr,
            "total_rounds": config.federated.num_rounds,
            "regularization": config.experiment.regularization
        }
    return fit_config_fn


def run_full_experiment(config_path: str):
    """
    Loads configuration, runs the full suite of FL experiments for all data families
    and instances, aggregates results, and saves them to a single file.
    """
    config = OmegaConf.load(config_path)
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- STARTING FULL EXPERIMENT RUN ---")
    print(f"Strategy: {config.experiment.strategy}, Regularization: {config.experiment.regularization}, Scenario: {config.experiment.scenario_flag}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment.strategy}_{config.experiment.regularization}_{config.experiment.scenario_flag}_{timestamp}"
    main_output_dir = os.path.join(config.path.output_path, run_name)
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Results for this run will be saved in: {main_output_dir}")

    # Create a sub-directory for intermediate files
    intermediate_path = os.path.join(main_output_dir, "intermediate_results")
    os.makedirs(intermediate_path, exist_ok=True)

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
    # Diffusion Model (if used)
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
    families = ['CF'] 
    all_results = {'individual_runs': [], 'family_histories': {}, 'overall_history': {}}
    
    # Loop over 4 families
    for family in families:
        family_histories = []
        print(f"\n--- Starting Family: {family} ---")
        
        # Loop over 10 instances in the family
        for i in range(1):
            print(f"-- Running Instance: {family}{i} --")
            
            # --- 3. RE-INITIALIZE FOR EACH RUN ---
            # This is critical to ensure each of the 40 runs is independent.
            
            # Load data for this specific instance
            vm_data = torch.tensor(np.load(f"{config.path.velocity_data_path}/{family}.npy")[i:i+1,:]).float()
            gt_seismic_data = torch.tensor(np.load(f"{config.path.gt_seismic_data_path}/{family}.npy")[i:i+1,:]).float().to(device)
            client_data_list = []
            for client_idx in range(config.experiment.num_clients):
                client_data = np.load(f"{config.path.client_seismic_data_path}/client{client_idx+1}/{family}.npy")[i:i+1,:]
                client_data_list.append(torch.tensor(client_data).float().to(device))

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
            strategy = strategy_class(
                fraction_fit=1.0, min_fit_clients=config.experiment.num_clients,
                min_available_clients=config.experiment.num_clients, evaluate_fn=evaluate_fn,
                fraction_evaluate=0.0, on_fit_config_fn=fit_config_fn,
                initial_parameters=ndarrays_to_parameters(tensor_to_ndarrays(initial_model.to(device))),
                server_momentum=config.experiment.server_momentum # Example param
            )
            
            # Create a fresh client function instance with the correct data partitions
            client_fn_instance = client_fn_factory(
                partitions=client_data_list, config=config, device=device,
                fwi_forward=fwi_forward, data_trans=data_trans, ssim_loss=ssim_loss,
                diffusion_args=diffusion_args,
                diffusion_state_dict=diffusion_state_dict
            )

            # --- 4. RUN ONE SIMULATION ---
            history = fl.simulation.start_simulation(
                client_fn=client_fn_instance,
                num_clients=config.experiment.num_clients,
                config=fl.server.ServerConfig(num_rounds=config.federated.num_rounds),
                strategy=strategy,
                client_resources={"num_gpus": 0.5} if device.type == "cuda" else {},
                ray_init_args={"include_dashboard": False}
            )

            # --- 5. STORE INDIVIDUAL RESULTS ---
            saved_ndarrays = final_parameters_store["final_model"]
            final_model = ndarrays_to_tensor(saved_ndarrays, device)
            
            run_result = {
                "family": family,
                "instance": i,
                "final_model": final_model.cpu().detach().numpy(),
                "history": history
            }
            all_results['individual_runs'].append(run_result)
            family_histories.append(history)

            intermediate_filename = os.path.join(intermediate_path, f"{family}_{i}_result.pkl")
            print(f"Saving intermediate result to: {intermediate_filename}")
            with open(intermediate_filename, 'wb') as f:
                pickle.dump(run_result, f)

        # --- 6. CALCULATE AND STORE FAMILY MEAN HISTORY ---
        all_results['family_histories'][family] = average_histories(family_histories)
        print(f"--- Finished Family: {family} ---")

    # --- 7. CALCULATE AND STORE OVERALL MEAN HISTORY ---
    all_individual_histories = [run['history'] for run in all_results['individual_runs']]
    all_results['overall_history'] = average_histories(all_individual_histories)
    
    # --- 8. SAVE THE FINAL AGGREGATED FILE ---
    # Save the final summary file inside the unique run directory
    final_filename = os.path.join(main_output_dir, "aggregated_results.pkl")
    with open(final_filename, 'wb') as f:
        pickle.dump(all_results, f)
        
    print(f"\n--- FULL EXPERIMENT COMPLETE ---")
    print(f"All results, including intermediates and the final aggregated file, are in: {main_output_dir}")