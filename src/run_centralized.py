import os
from omegaconf import OmegaConf
import numpy as np
from src.federated_learning.centralized_loss import *
import src.utils.data_trans as data_trans
import src.utils.pytorch_ssim as pytorch_ssim
from src.diffusion_models.diffusion_model import *
from src.pde_solvers.client_pde_solver import FWIForward
from src.full_waveform_inversion import *
from datetime import datetime
import glob
import torch
import pickle
def run_centralized(config_path: str, process_id: int, run_name = 'Centralized_Baseline', family = None, batch_size: str = None):
    #assert process_id is not None, "process_id is required"
    #assert process_id in [1,2], "process_id must be 1 (CF, CV) or 2 (FF, FV)"
    assert config_path is not None, "config_path is required"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    config = OmegaConf.load(config_path)
    gt_seismic_data_path = config["path"]["gt_seismic_data_path"]
    velocity_data_path = config["path"]["velocity_data_path"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir_name = f"{run_name}_{config.experiment.regularization}_{config.experiment.reg_lambda}_{config.experiment.scenario_flag}"
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

    model_path = config.path.model_path
    diffusion.load_state_dict(torch.load(model_path, map_location=device)['model'])
    diffusion.eval()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)  
    Inversion = run_inversion(diffusion, data_trans, pytorch_ssim, config.experiment.regularization)
    assert family is not None or process_id is not None, "Either family or process_id must be provided"
 
    # Determine which families to run
    if family is not None:
        families = [family]
    else:
        if process_id == 1:
            families = ['CF', 'CV']
        elif process_id == 2:
            families = ['FF', 'FV']
        else:
            raise ValueError("process_id must be 1 or 2")

    # Determine which instance indices to run (25 total per family)
    if family is not None and process_id is not None:
        # Split 25 instances into 12 (process 1) and 13 (process 2)
        instance_indices = list(range(12)) if process_id == 1 else list(range(12, 25))
    else:
        instance_indices = list(range(25))

    family_to_vm = {fam: np.load(f"{velocity_data_path}/{fam}.npy", mmap_mode="r") for fam in families}
    family_to_gt = {fam: np.load(f"{gt_seismic_data_path}/{fam}.npy", mmap_mode="r") for fam in families}

    for family in families:
        print(f"\n--- Starting Family: {family} ---")
        
        # Batch instances for speed (adjust batch_size per memory)
        # Resolve batch_size: 'max' -> entire partition; int string -> int; default to 2 if None
        if isinstance(batch_size, str):
            if batch_size.lower() == 'max':
                resolved_bs = len(instance_indices)
            else:
                try:
                    resolved_bs = int(batch_size)
                except Exception:
                    resolved_bs = 2
        elif isinstance(batch_size, int):
            resolved_bs = batch_size
        else:
            resolved_bs = 2
        for start in range(0, len(instance_indices), resolved_bs):
            idxs = instance_indices[start:start+resolved_bs]
            print(f"-- Running Instances: {family}{idxs} --")
            test_data = torch.from_numpy(family_to_gt[family][idxs, ...]).float().to(device)
            test_vm = torch.from_numpy(family_to_vm[family][idxs, ...]).float().to(device)
            test_init_vm = data_trans.prepare_initial_model(
                test_vm, initial_type="smoothed", sigma=config.forward.initial_sigma
            )
            test_init_vm = torch.nn.functional.pad(test_init_vm, (1, 1, 1, 1), mode="constant", value=0)
            ctx = {'n_grid': config.forward.n_grid, 'nt': config.forward.nt, 'dx': config.forward.dx, 
                    'nbc': config.forward.nbc, 'dt': config.forward.dt, 'f': config.forward.f,
                    'sz': config.forward.sz, 'gz': config.forward.gz, 'ng': config.forward.ng,
                    'ns': config.forward.ns}
            # Honor explicit source locations if provided (e.g., 2C)
            try:
                if hasattr(config.forward, 'sx') and config.forward.sx is not None:
                    ctx['sx'] = list(config.forward.sx)
            except Exception:
                pass

            fwi_forward = FWIForward(
                                    ctx, device, normalize=True, 
                                    v_denorm_func=data_trans.v_denormalize, 
                                    s_norm_func=data_trans.s_normalize_none
                                    )

            # Debug geometry summary for visibility
            try:
                print(f"Scenario: {config.experiment.scenario_flag}, ns (cfg): {config.forward.ns}, explicit sx: {getattr(config.forward, 'sx', None)}")
            except Exception:
                pass
            
            mu, final_results = Inversion.sample(
                                    mu = test_init_vm,
                                    mu_true=test_vm,
                                    y=test_data,
                                    ts=300,
                                    lr=0.03,
                                    reg_lambda=config.experiment.reg_lambda,
                                    fwi_forward=fwi_forward,
                                    scenario=config.experiment.scenario_flag,
                                    regularization=config.experiment.regularization,
                                    loss_type='l1'
                                    )  

            # Save each item in the batch
            try:
                # Compute per-sample final metrics to avoid batch-averaged summaries
                try:
                    mu_interior = mu[:, :, 1:-1, 1:-1]
                    predicted_seismic_full = fwi_forward(mu_interior, scenario=config.experiment.scenario_flag)
                except TypeError:
                    predicted_seismic_full = fwi_forward(mu_interior)
                regularization_method = Regularization_method(config.experiment.regularization, diffusion)
                per_sample_metrics = []
                for j in range(len(idxs)):
                    y_j = test_data[j:j+1]
                    pred_j = predicted_seismic_full[j:j+1]
                    obs_j = scenario_aware_seismic_loss(y_j, pred_j, config.experiment.scenario_flag)
                    reg_j = regularization_method.get_reg_loss(mu[j:j+1])
                    total_j = obs_j + config.experiment.reg_lambda * reg_j
                    vm_sample_unnorm = mu[j:j+1, :, 1:-1, 1:-1].detach().to('cpu')
                    vm_data_unnorm = data_trans.v_normalize(test_vm[j:j+1]).detach().to('cpu')
                    mae_j = l1(vm_sample_unnorm, vm_data_unnorm)
                    mse_j = l2(vm_sample_unnorm, vm_data_unnorm)
                    rmse_j = float(np.sqrt(mse_j.item()))
                    ssim_j = ssim_loss((vm_sample_unnorm + 1) / 2, (vm_data_unnorm + 1) / 2)
                    per_sample_metrics.append({
                        'obs_loss': float(obs_j.item()),
                        'reg_loss': float(reg_j.item()),
                        'total_loss': float(total_j.item()),
                        'mae': float(mae_j.item()),
                        'rmse': rmse_j,
                        'ssim': float(ssim_j.item()),
                    })
                mu_np = mu.detach().cpu().numpy()
                if mu_np.ndim == 4 and mu_np.shape[2] >= 3 and mu_np.shape[3] >= 3:
                    mu_np_int = mu_np[:, :, 1:-1, 1:-1]
                else:
                    mu_np_int = mu_np
                for j, idx in enumerate(idxs):
                    # Save pickle per instance
                    result_filename = os.path.join(main_output_dir, f"{family}_{idx}_result.pkl")
                    with open(result_filename, 'wb') as f:
                        pickle.dump({'final_results':final_results, 'metrics': per_sample_metrics[j], 'mu':mu_np[j]}, f)
                    # Save compact .npy per instance
                    npy_filename = os.path.join(main_output_dir, f"{family}_{idx}.npy")
                    np.save(npy_filename, np.ascontiguousarray(np.squeeze(mu_np_int[j])))
                print(f"Saved batch results for indices: {idxs}")
            except Exception as e:
                print(f"Warning: failed to save batch results for {family} {idxs}: {e}")
        print(f"--- Finished Family: {family} ---")
    print(f"\n--- EXPERIMENT COMPLETE ---")
    total_instances = len(families) * (len(instance_indices) if isinstance(instance_indices, list) else 25)
    print(f"Process {process_id} completed: {len(families)} families, {total_instances} instances")
    print(f"Individual results saved in: {main_output_dir}")


