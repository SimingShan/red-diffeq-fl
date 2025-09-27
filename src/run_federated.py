from datetime import datetime
import os
import glob
import pickle
from typing import List, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import torch
import src.utils.pytorch_ssim as pytorch_ssim
import torch.multiprocessing as mp
import flwr as fl
from flwr.common import (
    Metrics,
    ndarrays_to_parameters,
)
import torch.nn.functional as F
from flwr.server.strategy import FedAvg, FedAvgM, FedProx
from src.utils.data_trans import (
    s_normalize_none,
    v_denormalize,
)
import src.utils.data_trans as data_trans
from src.diffusion_models.diffusion_model import *  
from src.federated_learning.flwr_client import *                
from src.federated_learning.flwr_evaluation import get_evaluate_fn
from src.federated_learning.flwr_utils import *                 
from src.pde_solvers.client_pde_solver import FWIForward

def run_full_experiment(
    config_path: str,
    process_id: int,
    run_name: str,
    target_families: Optional[List[str]] = None,
    target_instances: Optional[List[int]] = None,
):
    """
    Loads configuration, runs the full suite of FL experiments for all data families
    and instances, aggregates results, and saves them to a single file.
    """
    assert process_id is not None, "process_id is required"
    assert process_id in [1,2], "process_id must be 1 (CF, CV) or 2 (FF, FV)"
    assert config_path is not None, "config_path is required"
    assert run_name in ['main', 'tuning', 'resume', 'rerun'], "run_name must be 'main', 'tuning', 'resume', or 'rerun'"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    config = OmegaConf.load(config_path)
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- STARTING FULL EXPERIMENT RUN ---")
    print(f"Strategy: {config.experiment.strategy}, Regularization: {config.experiment.regularization}, Scenario: {config.experiment.scenario_flag}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Directory naming (match existing convention with lambda before scenario)
    if run_name == "tuning":
        base_dir_name = (
            f"tuning_{config.experiment.strategy}_{config.experiment.regularization}_"
            f"{config.experiment.reg_lambda}_{config.experiment.server_momentum}_"
            f"{config.federated.local_lr}_{config.experiment.scenario_flag}"
        )
    else:
        base_dir_name = (
            f"main_{config.experiment.strategy}_{config.experiment.regularization}_"
            f"{config.experiment.reg_lambda}_{config.experiment.scenario_flag}"
        )
    
    # Find existing directories; prefer those with most instance results
    candidate_patterns = [
        os.path.join(config.path.output_path, f"{base_dir_name}_*"),
        # Older non-lambda naming
        os.path.join(config.path.output_path, f"main_{config.experiment.strategy}_{config.experiment.regularization}_{config.experiment.scenario_flag}_*"),
        # Generic lambda-aware pattern
        os.path.join(config.path.output_path, f"main_{config.experiment.strategy}_{config.experiment.regularization}_*_{config.experiment.scenario_flag}_*"),
    ]
    existing_dirs: List[str] = []
    seen = set()
    for pat in candidate_patterns:
        for p in glob.glob(pat):
            if p not in seen:
                seen.add(p)
                existing_dirs.append(p)
    if existing_dirs:
        def dir_score(d: str) -> Tuple[int, float]:
            count = len(glob.glob(os.path.join(d, "*_result.pkl")))
            return (count, os.path.getctime(d))
        main_output_dir = max(existing_dirs, key=dir_score)
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
    
    fit_config_fn = get_fit_config_fn(config)

    # --- 2. SETUP EXPERIMENT LOOP ---
    if process_id == 1:
        families = ['CF', 'CV']
    elif process_id == 2:
        families = ['FF', 'FV']
    # Optionally filter families
    if target_families is not None:
        families = [fam for fam in families if fam in set(target_families)]
    
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
        
        # Loop over instances in the family
        instance_indices = list(range(10)) if target_instances is None else list(target_instances)
        for i in instance_indices:
            print(f"-- Instance detected: {family}{i}", flush=True)
            # Instance-level resume: skip if result already exists (only in 'resume' mode)
            result_filename = os.path.join(main_output_dir, f"{family}_{i}_result.pkl")
            if run_name == 'resume' and os.path.exists(result_filename):
                print(f"   Skip (resume): {result_filename}", flush=True)
                continue
            print(f"   Run: {family}{i} -> {result_filename}", flush=True)
            
            # --- 3. RE-INITIALIZE FOR EACH RUN ---
            
            # Load data for this specific instance
            vm_np = family_to_vm[family][i:i+1, :]
            gt_np = family_to_gt[family][i:i+1, :]
            client_nps = [arr[i:i+1, :] for arr in family_to_clients[family]]

            vm_data = torch.from_numpy(np.ascontiguousarray(vm_np)).float()                     # CPU
            gt_seismic_data = torch.from_numpy(np.ascontiguousarray(gt_np)).float().pin_memory().to(device, non_blocking=True)
            client_data_list = [torch.from_numpy(np.ascontiguousarray(x)).float().pin_memory().to(device, non_blocking=True)
                                for x in client_nps]

            initial_model = data_trans.prepare_initial_model(vm_data, initial_type='smoothed', sigma=config.forward.initial_sigma)
            initial_model = F.pad(initial_model, (1, 1, 1, 1), "constant", 0)

            final_parameters_store = {}

            evaluate_fn = get_evaluate_fn(
                seismic_data=gt_seismic_data, mu_true=vm_data,
                fwi_forward=fwi_forward, data_trans=data_trans, ssim_loss=ssim_loss, device=device,
                total_rounds=config.federated.num_rounds,
                final_params_store=final_parameters_store, config=config,
                diffusion_model=server_diffusion_model
            )
            
            strategy_class = {"FedAvg": FedAvg, "FedAvgM": FedAvgM, "FedProx": FedProx}[config.experiment.strategy]
            
            strategy_params = {
                "fraction_fit": 1.0, 
                "min_fit_clients": config.experiment.num_clients,
                "min_available_clients": config.experiment.num_clients, 
                "evaluate_fn": evaluate_fn,
                "fraction_evaluate": 0.0, 
                "on_fit_config_fn": fit_config_fn,
                "initial_parameters": ndarrays_to_parameters(tensor_to_ndarrays(initial_model.to(device))),
            }
            
            if config.experiment.strategy == "FedAvgM":
                strategy_params["server_momentum"] = config.experiment.server_momentum
            elif config.experiment.strategy == "FedProx":
                # FedProx might need additional parameters
                pass
            
            strategy_params["fit_metrics_aggregation_fn"] = fit_metrics_fn
            
            strategy = strategy_class(**strategy_params)
            
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

            save_path = result_filename
            if run_name == 'rerun':
                # Compare to baseline SSIM if available; save as *_result_rerun.pkl only if improved
                baseline_ssim = None
                if os.path.exists(result_filename):
                    try:
                        with open(result_filename, 'rb') as bf:
                            baseline = pickle.load(bf)
                        base_hist = getattr(baseline.get('history', None), 'metrics_centralized', None)
                        if base_hist and 'ssim' in base_hist and len(base_hist['ssim']) > 0:
                            baseline_ssim = base_hist['ssim'][-1][1]
                    except Exception as e:
                        print(f"Warning: failed to read baseline SSIM from {result_filename}: {e}")

                final_ssim = None
                try:
                    hist_mc = getattr(history, 'metrics_centralized', None)
                    if hist_mc and 'ssim' in hist_mc and len(hist_mc['ssim']) > 0:
                        final_ssim = hist_mc['ssim'][-1][1]
                except Exception as e:
                    print(f"Warning: failed to read final SSIM for rerun comparison: {e}")

                if baseline_ssim is not None and final_ssim is not None and final_ssim > baseline_ssim:
                    save_path = os.path.join(main_output_dir, f"{family}_{i}_result_rerun.pkl")
                    print(f"Improved SSIM from {baseline_ssim:.6f} to {final_ssim:.6f}. Saving rerun to: {save_path}")
                else:
                    print("No improvement detected or baseline missing. Skipping save for rerun.")
                    save_path = None

            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump(run_result, f)
                    
        print(f"--- Finished Family: {family} ---")

    # --- 6. FINAL SUMMARY ---
    print(f"\n--- EXPERIMENT COMPLETE ---")
    print(f"Process {process_id} completed: {len(families)} families, {len(families) * 10} instances")
    print(f"Individual results saved in: {main_output_dir}")

    return all_results 